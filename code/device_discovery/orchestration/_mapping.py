"""把运行时设备观察转换成稳定的历史记录。"""

from __future__ import annotations

from device_discovery.collector import DeviceObservation
from device_discovery.history import DeviceHistoryRecord, DeviceHistoryResource


def build_history_record(
    matched_key: str, observation: DeviceObservation
) -> DeviceHistoryRecord:
    """由编排层已经确认的 Key 和观察结果构造持久化记录。"""
    resources = tuple(
        DeviceHistoryResource(
            subsystem=item.subsystem,
            system_path=item.system_path,
            name=item.name,
            device_type=item.device_type,
            device_node=item.device_node,
            driver=item.driver,
            interface_number=item.interface_number,
        )
        for item in observation.resources
    )
    return DeviceHistoryRecord(
        matched_key=matched_key,
        observed_at=observation.observed_at,
        source=observation.source.value,
        action=observation.action,
        usb_port_path=observation.usb_port_path,
        system_path=observation.system_path,
        device_path=observation.device_path,
        vid=observation.vid,
        pid=observation.pid,
        manufacturer=observation.manufacturer,
        product=observation.product,
        serial_number=observation.serial_number,
        bus_number=observation.bus_number,
        device_number=observation.device_number,
        driver=observation.driver,
        resources=resources,
        event_subsystem=observation.event_subsystem,
        event_system_path=observation.event_system_path,
        event_sequence_number=observation.event_sequence_number,
    )
