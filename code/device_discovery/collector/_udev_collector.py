"""基于 pyudev 的 Linux 系统设备采集器。"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .base import DeviceCollector, DeviceEventCallback
from .errors import (
    CollectorStateError,
    CollectorUnavailableError,
    DeviceMonitorError,
    DeviceScanError,
)
from .models import DeviceObservation, DeviceResource, ObservationSource

_LOGGER = logging.getLogger(__name__)
_USB_DEVICE_TYPE = "usb_device"
_USB_INTERFACE_TYPE = "usb_interface"


class UdevDeviceCollector(DeviceCollector):
    """使用 libudev 枚举 USB 设备并监听相关设备事件。"""

    def __init__(self, resource_subsystems: Iterable[str]) -> None:
        if not sys.platform.startswith("linux"):
            raise CollectorUnavailableError("设备采集器当前只支持 Linux")
        try:
            import pyudev
        except ImportError as exc:
            raise CollectorUnavailableError(
                "缺少 pyudev，请先安装 code/requirements.txt"
            ) from exc

        subsystems = tuple(dict.fromkeys(resource_subsystems))
        if not subsystems:
            raise ValueError("resource_subsystems 不能为空")

        self._pyudev = pyudev
        self._context = pyudev.Context()
        self._resource_subsystems = frozenset(subsystems)
        self._state_lock = threading.Lock()
        self._observer: Any | None = None
        self._callback: DeviceEventCallback | None = None

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            observer = self._observer
            return observer is not None and observer.is_alive()

    def scan(self) -> list[DeviceObservation]:
        """枚举非根集线器 USB 设备，并归并当前可见资源。"""
        observed_at = datetime.now(timezone.utc)
        try:
            devices = list(self._context.list_devices())
            roots = {
                device.sys_path: device
                for device in devices
                if self._is_physical_usb_device(device)
            }
            resources = self._group_resources(devices, roots)
            observations = [
                self._build_observation(
                    root,
                    observed_at=observed_at,
                    source=ObservationSource.SCAN,
                    action="present",
                    resources=resources.get(root.sys_path, ()),
                )
                for root in roots.values()
            ]
        except Exception as exc:
            if isinstance(exc, DeviceScanError):
                raise
            raise DeviceScanError("无法枚举当前 USB 设备") from exc
        return sorted(observations, key=lambda item: item.system_path)

    def start(self, callback: DeviceEventCallback) -> None:
        if not callable(callback):
            raise TypeError("callback 必须可调用")

        with self._state_lock:
            if self._observer is not None and self._observer.is_alive():
                raise CollectorStateError("设备监听已经启动")
            try:
                # 不设置单一子系统过滤器：USB 设备的 tty、net 和 usb_interface
                # 事件来自不同子系统，需要在回调中按 USB 父设备统一筛选。
                monitor = self._pyudev.Monitor.from_netlink(self._context)
                observer = self._pyudev.MonitorObserver(
                    monitor,
                    callback=self._handle_udev_event,
                    name="device-discovery-udev",
                )
                self._callback = callback
                self._observer = observer
                observer.start()
            except Exception as exc:
                self._callback = None
                self._observer = None
                raise DeviceMonitorError("无法启动 udev 设备监听") from exc

    def stop(self) -> None:
        with self._state_lock:
            observer = self._observer
            self._observer = None
            self._callback = None
        if observer is None:
            return
        try:
            observer.stop()
        except Exception as exc:
            raise DeviceMonitorError("无法停止 udev 设备监听") from exc

    def _handle_udev_event(self, device: Any) -> None:
        try:
            root = self._find_usb_root(device)
            if root is None or not self._is_relevant_event(device, root):
                return

            current_devices = list(self._context.list_devices())
            resources = list(self._resources_for_root(current_devices, root.sys_path))
            event_resource = self._build_resource_if_relevant(device)
            if event_resource is not None and all(
                item.system_path != event_resource.system_path for item in resources
            ):
                resources.append(event_resource)
            resources.sort(key=lambda item: item.system_path)

            observation = self._build_observation(
                root,
                observed_at=datetime.now(timezone.utc),
                source=ObservationSource.EVENT,
                action=device.action or "unknown",
                resources=tuple(resources),
                event_device=device,
            )
            with self._state_lock:
                callback = self._callback
            if callback is not None:
                callback(observation)
        except Exception:
            # 回调在线程中运行；记录单次转换或业务回调失败，保持后续监听。
            _LOGGER.exception("处理 udev 设备事件失败")

    def _group_resources(
        self, devices: list[Any], roots: dict[str, Any]
    ) -> dict[str, tuple[DeviceResource, ...]]:
        grouped: dict[str, list[DeviceResource]] = {path: [] for path in roots}
        for device in devices:
            resource = self._build_resource_if_relevant(device)
            if resource is None:
                continue
            root = self._find_usb_root(device)
            if root is not None and root.sys_path in grouped:
                grouped[root.sys_path].append(resource)
        return {
            path: tuple(sorted(items, key=lambda item: item.system_path))
            for path, items in grouped.items()
        }

    def _resources_for_root(
        self, devices: list[Any], root_path: str
    ) -> tuple[DeviceResource, ...]:
        resources: list[DeviceResource] = []
        for device in devices:
            resource = self._build_resource_if_relevant(device)
            if resource is None:
                continue
            root = self._find_usb_root(device)
            if root is not None and root.sys_path == root_path:
                resources.append(resource)
        return tuple(resources)

    def _is_physical_usb_device(self, device: Any) -> bool:
        return (
            device.subsystem == "usb"
            and device.device_type == _USB_DEVICE_TYPE
            # root hub 名称形如 usb1；外接设备名称形如 1-1、1-1.2。
            and "-" in device.sys_name
        )

    def _find_usb_root(self, device: Any) -> Any | None:
        if device.subsystem == "usb" and device.device_type == _USB_DEVICE_TYPE:
            return device if self._is_physical_usb_device(device) else None
        return device.find_parent("usb", _USB_DEVICE_TYPE)

    def _is_relevant_event(self, device: Any, root: Any) -> bool:
        if not self._is_physical_usb_device(root):
            return False
        return (
            device is root
            or device.subsystem in self._resource_subsystems
            or (
                device.subsystem == "usb"
                and device.device_type in {_USB_DEVICE_TYPE, _USB_INTERFACE_TYPE}
            )
        )

    def _build_resource_if_relevant(self, device: Any) -> DeviceResource | None:
        is_usb_interface = (
            device.subsystem == "usb" and device.device_type == _USB_INTERFACE_TYPE
        )
        if device.subsystem not in self._resource_subsystems and not is_usb_interface:
            return None
        return DeviceResource(
            subsystem=device.subsystem,
            system_path=device.sys_path,
            name=device.sys_name,
            device_type=device.device_type,
            device_node=device.device_node,
            driver=self._resource_driver(device),
            interface_number=self._text_property(device, "ID_USB_INTERFACE_NUM"),
        )

    def _resource_driver(self, device: Any) -> str | None:
        """返回资源实际使用的驱动；tty/net 通常继承自 USB 接口。"""
        if device.driver:
            return device.driver
        driver = self._text_property(device, "ID_USB_DRIVER")
        if driver:
            return driver
        interface = device.find_parent("usb", _USB_INTERFACE_TYPE)
        return None if interface is None else interface.driver

    def _build_observation(
        self,
        root: Any,
        *,
        observed_at: datetime,
        source: ObservationSource,
        action: str,
        resources: tuple[DeviceResource, ...],
        event_device: Any | None = None,
    ) -> DeviceObservation:
        return DeviceObservation(
            observed_at=observed_at,
            source=source,
            action=action,
            usb_port_path=root.sys_name,
            system_path=root.sys_path,
            device_path=root.device_path,
            vid=self._usb_id(root, "ID_VENDOR_ID", "idVendor"),
            pid=self._usb_id(root, "ID_MODEL_ID", "idProduct"),
            manufacturer=self._first_text(
                root, ("ID_VENDOR_FROM_DATABASE", "ID_VENDOR"), "manufacturer"
            ),
            product=self._first_text(
                root, ("ID_MODEL_FROM_DATABASE", "ID_MODEL"), "product"
            ),
            serial_number=self._first_text(root, ("ID_SERIAL_SHORT",), "serial"),
            bus_number=self._attribute_text(root, "busnum"),
            device_number=self._attribute_text(root, "devnum"),
            driver=root.driver,
            resources=resources,
            event_subsystem=None if event_device is None else event_device.subsystem,
            event_system_path=None if event_device is None else event_device.sys_path,
            event_sequence_number=self._sequence_number(event_device),
        )

    def _usb_id(
        self, device: Any, property_name: str, attribute_name: str
    ) -> str | None:
        value = self._text_property(device, property_name)
        if value is None:
            value = self._attribute_text(device, attribute_name)
        return None if value is None else value.lower().removeprefix("0x").zfill(4)

    def _first_text(
        self, device: Any, property_names: tuple[str, ...], attribute_name: str
    ) -> str | None:
        for property_name in property_names:
            value = self._text_property(device, property_name)
            if value is not None:
                return value
        return self._attribute_text(device, attribute_name)

    @staticmethod
    def _text_property(device: Any, name: str) -> str | None:
        value = device.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _attribute_text(device: Any, name: str) -> str | None:
        try:
            value = device.attributes.get(name)
        except (KeyError, OSError):
            return None
        if value is None:
            return None
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace").strip()
        else:
            text = str(value).strip()
        return text or None

    @staticmethod
    def _sequence_number(device: Any | None) -> int | None:
        if device is None:
            return None
        try:
            return int(device.sequence_number)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
