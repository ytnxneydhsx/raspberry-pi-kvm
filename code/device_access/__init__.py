"""USB 多资源接入与通用串口能力的公开入口。"""

from __future__ import annotations

from pathlib import Path

from device_discovery.orchestration import RecognizedDevice

from ._config import load_profile_path
from ._profiles import JsonSerialProfileStore
from ._usb import UsbDeviceAccess
from .base import SerialChannel
from .errors import (
    ChannelStateError,
    DeviceAccessError,
    InvalidSerialConfigError,
    ProfileStoreError,
    ResourceUnavailableError,
    SerialConnectionError,
    SerialWriteTimeoutError,
)
from .models import ChannelState, ConfigField, SerialConfig, get_serial_config_fields


def create_usb_device(result: RecognizedDevice) -> UsbDeviceAccess:
    """接收发现模块已经匹配的结果，不自动打开或修改任何设备。"""
    return UsbDeviceAccess(result)


def create_profile_store(
    path: str | Path | None = None, *, config_path: str | Path | None = None
) -> JsonSerialProfileStore:
    """创建方案仓库；显式 initialize() 才会准备存储文件。"""
    return JsonSerialProfileStore(path if path is not None else load_profile_path(config_path))


__all__ = [
    "ChannelState", "ChannelStateError", "ConfigField", "DeviceAccessError",
    "InvalidSerialConfigError", "JsonSerialProfileStore", "ProfileStoreError",
    "ResourceUnavailableError", "SerialChannel", "SerialConfig",
    "SerialConnectionError", "SerialWriteTimeoutError", "UsbDeviceAccess",
    "create_profile_store", "create_usb_device", "get_serial_config_fields",
]
