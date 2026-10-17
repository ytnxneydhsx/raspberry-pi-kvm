"""系统设备采集模块对外使用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ObservationSource(str, Enum):
    """设备观察的来源。"""

    SCAN = "scan"
    EVENT = "event"


@dataclass(frozen=True)
class DeviceResource:
    """USB 设备在 Linux 中呈现的一项接口或访问入口。"""

    subsystem: str
    system_path: str
    name: str
    device_type: str | None = None
    device_node: str | None = None
    driver: str | None = None
    interface_number: str | None = None


@dataclass(frozen=True)
class DeviceObservation:
    """一次扫描或 udev 事件产生的 USB 设备观察结果。"""

    observed_at: datetime
    source: ObservationSource
    action: str
    usb_port_path: str
    system_path: str
    device_path: str
    vid: str | None
    pid: str | None
    manufacturer: str | None = None
    product: str | None = None
    serial_number: str | None = None
    bus_number: str | None = None
    device_number: str | None = None
    driver: str | None = None
    resources: tuple[DeviceResource, ...] = ()
    event_subsystem: str | None = None
    event_system_path: str | None = None
    event_sequence_number: int | None = None
