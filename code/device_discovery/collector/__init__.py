"""系统设备采集模块的稳定公开入口。"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path

from ._config import load_resource_subsystems
from ._udev_collector import UdevDeviceCollector
from .base import DeviceCollector, DeviceEventCallback
from .errors import (
    CollectorConfigurationError,
    CollectorStateError,
    CollectorUnavailableError,
    DeviceCollectorError,
    DeviceMonitorError,
    DeviceScanError,
)
from .models import DeviceObservation, DeviceResource, ObservationSource

_default_collector: DeviceCollector | None = None
_default_collector_lock = threading.Lock()


def create_device_collector(
    resource_subsystems: Iterable[str] | None = None,
    *,
    config_path: str | Path | None = None,
) -> DeviceCollector:
    """创建采集器；未传子系统时从 application.yaml 读取。"""
    subsystems = (
        tuple(resource_subsystems)
        if resource_subsystems is not None
        else load_resource_subsystems(config_path)
    )
    return UdevDeviceCollector(subsystems)


def get_device_collector() -> DeviceCollector:
    """返回当前进程共享的默认采集器。"""
    global _default_collector
    if _default_collector is None:
        with _default_collector_lock:
            if _default_collector is None:
                _default_collector = create_device_collector()
    return _default_collector


__all__ = [
    "CollectorConfigurationError",
    "CollectorStateError",
    "CollectorUnavailableError",
    "DeviceCollector",
    "DeviceCollectorError",
    "DeviceEventCallback",
    "DeviceMonitorError",
    "DeviceObservation",
    "DeviceResource",
    "DeviceScanError",
    "ObservationSource",
    "create_device_collector",
    "get_device_collector",
]
