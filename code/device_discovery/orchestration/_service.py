"""规则、设备采集与历史保存的最小业务编排。"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import Enum

from device_discovery.collector import DeviceCollector, DeviceObservation
from device_discovery.history import DeviceHistoryStore
from device_discovery.rules import Rule, RuleStore

from ._mapping import build_history_record
from .errors import (
    OrchestratorScanError,
    OrchestratorStartError,
    OrchestratorStateError,
    OrchestratorStopError,
)
from .models import (
    FailureStage,
    IgnoreReason,
    IgnoredDevice,
    ProcessingFailure,
    ProcessingOutcome,
    RecognizedDevice,
    RuleChangeReport,
    ScanReport,
)

_LOGGER = logging.getLogger(__name__)

ResultCallback = Callable[[RecognizedDevice], None]
FailureCallback = Callable[[ProcessingFailure], None]


class _LifecycleState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


class DeviceDiscoveryOrchestrator:
    """实现发现、匹配、保存、成功后输出的调用顺序。"""

    def __init__(
        self,
        rule_store: RuleStore,
        collector: DeviceCollector,
        history_store: DeviceHistoryStore,
    ) -> None:
        self._rule_store = rule_store
        self._collector = collector
        self._history_store = history_store
        self._state_lock = threading.Lock()
        self._processing_lock = threading.RLock()
        self._rule_change_lock = threading.Lock()
        self._state = _LifecycleState.STOPPED
        self._result_callback: ResultCallback | None = None
        self._failure_callback: FailureCallback | None = None

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._state is _LifecycleState.RUNNING

    def start(
        self,
        result_callback: ResultCallback | None = None,
        failure_callback: FailureCallback | None = None,
    ) -> ScanReport:
        """先建立增量监听，再扫描当前设备并返回启动扫描报告。"""
        self._validate_callback(result_callback, "result_callback")
        self._validate_callback(failure_callback, "failure_callback")
        with self._state_lock:
            if self._state is not _LifecycleState.STOPPED:
                raise OrchestratorStateError(
                    f"编排器当前状态不允许启动：{self._state.value}"
                )
            self._state = _LifecycleState.STARTING
            self._result_callback = result_callback
            self._failure_callback = failure_callback

        try:
            self._collector.start(self._handle_collector_event)
        except Exception as exc:
            self._reset_stopped_state()
            raise OrchestratorStartError("无法启动设备变化监听") from exc

        with self._state_lock:
            self._state = _LifecycleState.RUNNING

        try:
            return self.scan()
        except OrchestratorScanError as exc:
            try:
                self._collector.stop()
            except Exception:
                _LOGGER.exception("启动扫描失败后停止采集器时再次失败")
            self._reset_stopped_state()
            raise OrchestratorStartError("监听已建立，但启动扫描失败") from exc

    def stop(self) -> None:
        """停止增量监听；已停止时重复调用视为成功。"""
        with self._state_lock:
            if self._state is _LifecycleState.STOPPED:
                return
            if self._state is _LifecycleState.STOPPING:
                raise OrchestratorStateError("编排器正在停止")
            self._state = _LifecycleState.STOPPING

        try:
            self._collector.stop()
        except Exception as exc:
            self._reset_stopped_state()
            raise OrchestratorStopError("无法停止设备变化监听") from exc
        self._reset_stopped_state()

    def scan(self) -> ScanReport:
        """扫描、分类、保存匹配结果，并在保存成功后发送回调。"""
        outcomes = self._collect_scan_outcomes()
        return self._build_scan_report(outcomes)

    def _collect_scan_outcomes(self) -> tuple[ProcessingOutcome, ...]:
        """在处理锁内完成扫描、匹配和历史追加。"""
        with self._processing_lock:
            try:
                observations = self._collector.scan()
            except Exception as exc:
                raise OrchestratorScanError("无法扫描当前设备") from exc
            return tuple(self._process(item) for item in observations)

    def _build_scan_report(
        self, outcomes: tuple[ProcessingOutcome, ...]
    ) -> ScanReport:
        """在内部锁之外发送回调并组织扫描报告。"""
        recognized: list[RecognizedDevice] = []
        ignored: list[IgnoredDevice] = []
        failures: list[ProcessingFailure] = []
        for outcome in outcomes:
            if isinstance(outcome, RecognizedDevice):
                recognized.append(outcome)
                callback_failure = self._send_result(outcome)
                if callback_failure is not None:
                    failures.append(callback_failure)
                    self._send_failure(callback_failure)
            elif isinstance(outcome, IgnoredDevice):
                ignored.append(outcome)
            else:
                failures.append(outcome)
                self._send_failure(outcome)
        return ScanReport(tuple(recognized), tuple(ignored), tuple(failures))

    def add_rule(self, key: str, vid: str, pid: str) -> RuleChangeReport:
        """保存新规则，然后立即重新扫描当前设备。"""
        with self._rule_change_lock:
            rule = self._rule_store.add(key, vid, pid)
            outcomes, scan_error = self._scan_after_rule_change()
        return self._build_rule_change_report(rule, outcomes, scan_error)

    def update_rule(
        self, key: str, vid: str, pid: str, *, new_key: str | None = None
    ) -> RuleChangeReport:
        """保存规则修改，然后立即重新扫描当前设备。"""
        with self._rule_change_lock:
            rule = self._rule_store.update(key, vid, pid, new_key=new_key)
            outcomes, scan_error = self._scan_after_rule_change()
        return self._build_rule_change_report(rule, outcomes, scan_error)

    def delete_rule(self, key: str) -> RuleChangeReport:
        """删除规则，然后立即重新扫描当前设备。"""
        with self._rule_change_lock:
            rule = self._rule_store.delete(key)
            outcomes, scan_error = self._scan_after_rule_change()
        return self._build_rule_change_report(rule, outcomes, scan_error)

    def _process(self, observation: DeviceObservation) -> ProcessingOutcome:
        with self._processing_lock:
            if observation.vid is None or observation.pid is None:
                return IgnoredDevice(IgnoreReason.MISSING_USB_ID, observation)

            try:
                matched_key = self._rule_store.match(observation.vid, observation.pid)
            except Exception as exc:
                return ProcessingFailure(FailureStage.RULE_MATCH, observation, exc)
            if matched_key is None:
                return IgnoredDevice(IgnoreReason.UNMATCHED, observation)

            try:
                history_record = build_history_record(matched_key, observation)
            except Exception as exc:
                return ProcessingFailure(FailureStage.HISTORY_BUILD, observation, exc)

            try:
                self._history_store.append(history_record)
            except Exception as exc:
                return ProcessingFailure(FailureStage.HISTORY_APPEND, observation, exc)
            return RecognizedDevice(matched_key, observation)

    def _handle_collector_event(self, observation: DeviceObservation) -> None:
        """运行在采集器回调线程中，异常不得逃逸并终止监听。"""
        try:
            with self._state_lock:
                if self._state not in {
                    _LifecycleState.STARTING,
                    _LifecycleState.RUNNING,
                }:
                    return
            outcome = self._process(observation)
            if isinstance(outcome, RecognizedDevice):
                callback_failure = self._send_result(outcome)
                if callback_failure is not None:
                    self._send_failure(callback_failure)
            elif isinstance(outcome, ProcessingFailure):
                self._send_failure(outcome)
        except Exception:
            _LOGGER.exception("处理设备变化事件时发生未预期错误")

    def _send_result(
        self, result: RecognizedDevice
    ) -> ProcessingFailure | None:
        with self._state_lock:
            callback = self._result_callback
        if callback is None:
            return None
        try:
            callback(result)
        except Exception as exc:
            return ProcessingFailure(
                FailureStage.RESULT_CALLBACK, result.observation, exc
            )
        return None

    def _send_failure(self, failure: ProcessingFailure) -> None:
        with self._state_lock:
            callback = self._failure_callback
        if callback is None:
            _LOGGER.error(
                "设备观察处理失败：stage=%s path=%s",
                failure.stage.value,
                failure.observation.system_path,
                exc_info=(
                    type(failure.error),
                    failure.error,
                    failure.error.__traceback__,
                ),
            )
            return
        try:
            callback(failure)
        except Exception:
            _LOGGER.exception("设备处理失败回调自身发生异常")

    def _scan_after_rule_change(
        self,
    ) -> tuple[tuple[ProcessingOutcome, ...] | None, OrchestratorScanError | None]:
        try:
            return self._collect_scan_outcomes(), None
        except OrchestratorScanError as exc:
            return None, exc

    def _build_rule_change_report(
        self,
        rule: Rule,
        outcomes: tuple[ProcessingOutcome, ...] | None,
        scan_error: OrchestratorScanError | None,
    ) -> RuleChangeReport:
        if scan_error is not None:
            return RuleChangeReport(rule=rule, scan=None, scan_error=scan_error)
        assert outcomes is not None
        return RuleChangeReport(
            rule=rule,
            scan=self._build_scan_report(outcomes),
            scan_error=None,
        )

    def _reset_stopped_state(self) -> None:
        with self._state_lock:
            self._state = _LifecycleState.STOPPED
            self._result_callback = None
            self._failure_callback = None

    @staticmethod
    def _validate_callback(callback: object, name: str) -> None:
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} 必须可调用或为 None")
