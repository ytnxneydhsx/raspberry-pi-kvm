"""设备发现编排层对外使用的结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from device_discovery.collector import DeviceObservation
from device_discovery.rules import Rule


class IgnoreReason(str, Enum):
    """设备观察结果未进入正常输出的原因。"""

    MISSING_USB_ID = "missing_usb_id"
    UNMATCHED = "unmatched"


class FailureStage(str, Enum):
    """单条设备观察处理失败的阶段。"""

    RULE_MATCH = "rule_match"
    HISTORY_BUILD = "history_build"
    HISTORY_APPEND = "history_append"
    RESULT_CALLBACK = "result_callback"


@dataclass(frozen=True)
class RecognizedDevice:
    """已经匹配、保存并可向外提供的设备结果。"""

    matched_key: str
    observation: DeviceObservation


@dataclass(frozen=True)
class IgnoredDevice:
    """因身份缺失或未匹配关注规则而忽略的设备观察。"""

    reason: IgnoreReason
    observation: DeviceObservation


@dataclass(frozen=True)
class ProcessingFailure:
    """单条设备观察在编排流程中的失败信息。"""

    stage: FailureStage
    observation: DeviceObservation
    error: Exception


ProcessingOutcome = RecognizedDevice | IgnoredDevice | ProcessingFailure


@dataclass(frozen=True)
class ScanReport:
    """一次全量扫描的分类结果。"""

    recognized: tuple[RecognizedDevice, ...]
    ignored: tuple[IgnoredDevice, ...]
    failures: tuple[ProcessingFailure, ...]


@dataclass(frozen=True)
class RuleChangeReport:
    """规则已经保存后的重新扫描结果。"""

    rule: Rule
    scan: ScanReport | None
    scan_error: Exception | None
