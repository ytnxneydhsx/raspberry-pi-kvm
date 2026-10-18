"""把已有发现结果组织成一个 USB 设备及其多个资源入口。"""

from __future__ import annotations

import threading
from pathlib import Path

from device_discovery.collector import DeviceObservation, DeviceResource
from device_discovery.orchestration import RecognizedDevice

from ._serial import PySerialChannel
from .base import SerialChannel
from .errors import ResourceUnavailableError
from .models import SerialConfig


class UsbDeviceAccess:
    """用户态能力容器；读取元数据和创建连接对象均不会打开串口。"""

    def __init__(self, result: RecognizedDevice) -> None:
        if not isinstance(result, RecognizedDevice):
            raise TypeError("result 必须来自发现模块的 RecognizedDevice")
        self._lock = threading.RLock()
        self._result = result
        self._present = not self._root_removed(result.observation)
        self._resources = self._effective_resources(result.observation)
        self._channels: dict[str, PySerialChannel] = {}

    @property
    def matched_key(self) -> str:
        with self._lock:
            return self._result.matched_key

    @property
    def observation(self) -> DeviceObservation:
        with self._lock:
            return self._result.observation

    @property
    def is_present(self) -> bool:
        with self._lock:
            return self._present

    @property
    def resources(self) -> tuple[DeviceResource, ...]:
        with self._lock:
            return self._resources

    @property
    def serial_resources(self) -> tuple[DeviceResource, ...]:
        with self._lock:
            return tuple(item for item in self._resources
                         if item.subsystem == "tty" and item.device_node)

    @property
    def network_resources(self) -> tuple[DeviceResource, ...]:
        with self._lock:
            return tuple(item for item in self._resources if item.subsystem == "net")

    def serial_channel(
        self, resource_path: str, config: SerialConfig | None = None
    ) -> SerialChannel:
        """按所选资源的 system_path 获取独立串口连接，显式 open() 才通信。"""
        if config is not None and not isinstance(config, SerialConfig):
            raise TypeError("config 必须是 SerialConfig 或 None")
        with self._lock:
            resource = self._find_serial(resource_path)
            channel = self._channels.get(resource_path)
            if channel is not None and channel.resource.device_node == resource.device_node:
                if config is not None and config != channel.config:
                    channel.configure(config)
                return channel

            def ensure_available() -> None:
                current = self._find_serial(resource_path)
                if self._channels.get(resource_path) is not channel:
                    raise ResourceUnavailableError("该连接对象已经被替换，请重新获取连接")
                if current.device_node != resource.device_node:
                    raise ResourceUnavailableError("串口入口已经变化，请重新选择资源")
                self._check_system_resource(current)

            channel = PySerialChannel(
                resource, config or SerialConfig(),
                ensure_available=ensure_available, lock=self._lock,
            )
            self._channels[resource_path] = channel
            return channel

    def update(self, result: RecognizedDevice) -> None:
        """由调用方转交发现事件；更新资源并关闭消失的连接，不自动连接。"""
        if not isinstance(result, RecognizedDevice):
            raise TypeError("result 必须是 RecognizedDevice")
        with self._lock:
            previous = self._result.observation
            current = result.observation
            if current.system_path != previous.system_path:
                raise ResourceUnavailableError("该观察结果属于另一个 USB 设备位置")
            if not self._present:
                raise ResourceUnavailableError("旧 USB 设备已经移除，请从新的发现结果重新创建对象")
            if ((current.vid, current.pid) != (previous.vid, previous.pid)
                    or (current.serial_number and previous.serial_number
                        and current.serial_number != previous.serial_number)
                    or any(old is not None and new is not None and old != new
                           for old, new in (
                               (previous.bus_number, current.bus_number),
                               (previous.device_number, current.device_number),
                           ))):
                self._present = False
                self._resources = ()
                self._disconnect_all("USB 身份或枚举编号已变化，请重新选择设备")
                raise ResourceUnavailableError("USB 身份或枚举编号已变化，旧设备对象已失效")
            self._result = result
            self._present = not self._root_removed(current)
            self._resources = self._effective_resources(current)
            available = {item.system_path: item for item in self.serial_resources}
            for path, channel in self._channels.items():
                resource = available.get(path)
                if resource is None or resource.device_node != channel.resource.device_node:
                    channel.mark_disconnected("串口资源已经移除或入口变化，请手动重新连接")
                else:
                    channel.refresh_resource(resource)

    def close(self) -> None:
        """释放此设备已经创建的全部串口连接。"""
        with self._lock:
            failures: list[Exception] = []
            for channel in self._channels.values():
                try:
                    channel.close()
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise failures[0]

    def _find_serial(self, resource_path: str) -> DeviceResource:
        if not self._present:
            raise ResourceUnavailableError("USB 设备已移除")
        for item in self.serial_resources:
            if item.system_path == resource_path:
                return item
        raise ResourceUnavailableError("当前设备没有所选的串口资源")

    def _disconnect_all(self, reason: str) -> None:
        for channel in self._channels.values():
            channel.mark_disconnected(reason)

    def _check_system_resource(self, resource: DeviceResource) -> None:
        """打开前只读复核 sysfs，拒绝已消失或被重新枚举的旧入口。"""
        observation = self._result.observation
        root = Path(observation.system_path)
        resource_path = Path(resource.system_path)
        if root not in resource_path.parents:
            raise ResourceUnavailableError("串口资源不属于这个 USB 设备")
        if not resource_path.exists() or not resource.device_node or not Path(resource.device_node).exists():
            raise ResourceUnavailableError("串口系统入口已消失，请刷新发现结果")
        expected = {
            "idVendor": observation.vid,
            "idProduct": observation.pid,
            "serial": observation.serial_number,
            "busnum": observation.bus_number,
            "devnum": observation.device_number,
        }
        try:
            for attribute, value in expected.items():
                if value is not None and (root / attribute).read_text().strip() != value:
                    raise ResourceUnavailableError("USB 身份或枚举编号已变化，请刷新发现结果")
        except OSError as exc:
            raise ResourceUnavailableError("无法确认 USB 系统资源，请刷新发现结果") from exc

    @staticmethod
    def _root_removed(observation: DeviceObservation) -> bool:
        return (observation.action == "remove"
                and observation.event_system_path == observation.system_path)

    @classmethod
    def _effective_resources(cls, observation: DeviceObservation) -> tuple[DeviceResource, ...]:
        if cls._root_removed(observation):
            return ()
        removed_path = (observation.event_system_path
                        if observation.action == "remove" else None)
        if not removed_path:
            return observation.resources
        # 发现模块可能保留触发移除的资源用于历史；这里从可操作入口中排除它。
        return tuple(item for item in observation.resources
                     if item.system_path != removed_path
                     and not item.system_path.startswith(removed_path.rstrip("/") + "/"))
