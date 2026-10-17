"""设备连接历史模块对外使用的持久化数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DeviceHistoryResource:
    """写入历史的一项设备接口或访问入口。"""

    subsystem: str
    system_path: str
    name: str
    device_type: str | None = None
    device_node: str | None = None
    driver: str | None = None
    interface_number: str | None = None


@dataclass(frozen=True)
class DeviceHistoryRecord:
    """由编排层组织并交给历史模块保存的完整设备记录。"""

    matched_key: str
    observed_at: datetime
    source: str
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
    resources: tuple[DeviceHistoryResource, ...] = ()
    event_subsystem: str | None = None
    event_system_path: str | None = None
    event_sequence_number: int | None = None
