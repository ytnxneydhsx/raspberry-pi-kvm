"""设备连接历史记录的 JSON Lines 序列化。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .errors import InvalidHistoryRecordError
from .models import DeviceHistoryRecord, DeviceHistoryResource

_SCHEMA_VERSION = 1


def serialize_record(record: DeviceHistoryRecord) -> bytes:
    """验证记录并生成一行 UTF-8 JSON，结尾包含换行符。"""
    if not isinstance(record, DeviceHistoryRecord):
        raise InvalidHistoryRecordError("record 必须是 DeviceHistoryRecord")
    _require_text(record.matched_key, "matched_key")
    _require_text(record.source, "source")
    _require_text(record.action, "action")
    _require_text(record.usb_port_path, "usb_port_path")
    _require_text(record.system_path, "system_path")
    _require_text(record.device_path, "device_path")
    _optional_text(record.vid, "vid")
    _optional_text(record.pid, "pid")
    _optional_text(record.manufacturer, "manufacturer")
    _optional_text(record.product, "product")
    _optional_text(record.serial_number, "serial_number")
    _optional_text(record.bus_number, "bus_number")
    _optional_text(record.device_number, "device_number")
    _optional_text(record.driver, "driver")
    _optional_text(record.event_subsystem, "event_subsystem")
    _optional_text(record.event_system_path, "event_system_path")
    _optional_integer(record.event_sequence_number, "event_sequence_number")
    if not isinstance(record.observed_at, datetime):
        raise InvalidHistoryRecordError("observed_at 必须是 datetime")
    if record.observed_at.tzinfo is None or record.observed_at.utcoffset() is None:
        raise InvalidHistoryRecordError("observed_at 必须包含时区")

    if not isinstance(record.resources, tuple):
        raise InvalidHistoryRecordError("resources 必须是 tuple")
    resources = [_resource_to_dict(item) for item in record.resources]
    document = {
        "schema_version": _SCHEMA_VERSION,
        "matched_key": record.matched_key,
        "observed_at": _format_datetime(record.observed_at),
        "source": record.source,
        "action": record.action,
        "usb_port_path": record.usb_port_path,
        "system_path": record.system_path,
        "device_path": record.device_path,
        "vid": record.vid,
        "pid": record.pid,
        "manufacturer": record.manufacturer,
        "product": record.product,
        "serial_number": record.serial_number,
        "bus_number": record.bus_number,
        "device_number": record.device_number,
        "driver": record.driver,
        "resources": resources,
        "event_subsystem": record.event_subsystem,
        "event_system_path": record.event_system_path,
        "event_sequence_number": record.event_sequence_number,
    }
    try:
        line = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidHistoryRecordError(f"历史记录无法序列化：{exc}") from exc
    return (line + "\n").encode("utf-8")


def _resource_to_dict(resource: DeviceHistoryResource) -> dict[str, object]:
    if not isinstance(resource, DeviceHistoryResource):
        raise InvalidHistoryRecordError(
            "resources 中的每一项都必须是 DeviceHistoryResource"
        )
    _require_text(resource.subsystem, "resource.subsystem")
    _require_text(resource.system_path, "resource.system_path")
    _require_text(resource.name, "resource.name")
    _optional_text(resource.device_type, "resource.device_type")
    _optional_text(resource.device_node, "resource.device_node")
    _optional_text(resource.driver, "resource.driver")
    _optional_text(resource.interface_number, "resource.interface_number")
    return {
        "subsystem": resource.subsystem,
        "system_path": resource.system_path,
        "name": resource.name,
        "device_type": resource.device_type,
        "device_node": resource.device_node,
        "driver": resource.driver,
        "interface_number": resource.interface_number,
    }


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidHistoryRecordError(f"{field_name} 必须是非空字符串")


def _optional_text(value: object, field_name: str) -> None:
    if value is not None:
        _require_text(value, field_name)


def _optional_integer(value: object, field_name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise InvalidHistoryRecordError(f"{field_name} 必须是整数或 None")


def _format_datetime(value: datetime) -> str:
    try:
        utc_value = value.astimezone(timezone.utc)
    except (AttributeError, OverflowError, ValueError) as exc:
        raise InvalidHistoryRecordError("observed_at 不是有效时间") from exc
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
