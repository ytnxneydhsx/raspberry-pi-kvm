"""串口配置、通用配置项描述和连接状态。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .errors import InvalidSerialConfigError


class ChannelState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True)
class SerialConfig:
    """用户选择的主机侧参数，不包含端口或板卡身份。"""

    baudrate: int = 115200
    bytesize: int = 8
    parity: str = "N"
    stopbits: float = 1
    flow_control: str = "none"
    timeout: float = 0.5
    write_timeout: float = 2.0
    dtr: bool = False
    rts: bool = False

    def __post_init__(self) -> None:
        if type(self.baudrate) is not int or self.baudrate <= 0:
            raise InvalidSerialConfigError("baudrate 必须是正整数")
        if type(self.bytesize) is not int or self.bytesize not in (5, 6, 7, 8):
            raise InvalidSerialConfigError("bytesize 必须是 5、6、7 或 8")
        if not isinstance(self.parity, str) or self.parity not in ("N", "E", "O", "M", "S"):
            raise InvalidSerialConfigError("parity 必须是 N、E、O、M 或 S")
        if type(self.stopbits) not in (int, float) or self.stopbits not in (1, 1.5, 2):
            raise InvalidSerialConfigError("stopbits 必须是 1、1.5 或 2")
        if not isinstance(self.flow_control, str) or self.flow_control not in (
            "none", "rtscts", "xonxoff"
        ):
            raise InvalidSerialConfigError("flow_control 必须是 none、rtscts 或 xonxoff")
        for name in ("timeout", "write_timeout"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise InvalidSerialConfigError(f"{name} 必须是有限的非负秒数")
        for name in ("dtr", "rts"):
            if type(getattr(self, name)) is not bool:
                raise InvalidSerialConfigError(f"{name} 必须是布尔值")


@dataclass(frozen=True)
class ConfigField:
    """应用提供的配置说明；choices 不代表实机完整支持清单。"""

    name: str
    value_type: str
    default: str | int | float | bool
    choices: tuple[str | int | float | bool, ...] = ()
    allow_custom: bool = False
    description: str = ""


_DRIVER_OPTIONS_PATH = Path(__file__).with_name("serial_driver_options.json")


def get_serial_config_fields(driver: str | None = None) -> tuple[ConfigField, ...]:
    """按 Linux 驱动名读取应用维护的配置选项，不打开或探测设备。"""
    try:
        with _DRIVER_OPTIONS_PATH.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取串口驱动配置：{_DRIVER_OPTIONS_PATH}") from exc

    if document.get("version") != 1 or not isinstance(document.get("drivers"), dict):
        raise RuntimeError("串口驱动配置格式无效")

    definitions = document["drivers"]
    selected = definitions.get(driver) if driver else None
    if selected is None:
        selected = definitions.get("default")
    if not isinstance(selected, dict) or not isinstance(selected.get("parameters"), dict):
        raise RuntimeError(f"串口驱动配置缺少可用定义：{driver or 'default'}")

    fields: list[ConfigField] = []
    for name, raw in selected["parameters"].items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise RuntimeError("串口驱动参数定义无效")
        choices = raw.get("choices", [])
        if not isinstance(choices, list):
            raise RuntimeError(f"串口驱动参数 choices 必须是数组：{name}")
        try:
            fields.append(ConfigField(
                name=name,
                value_type=raw["type"],
                default=raw["default"],
                choices=tuple(choices),
                allow_custom=raw.get("allow_custom", False),
                description=raw.get("description", ""),
            ))
        except KeyError as exc:
            raise RuntimeError(f"串口驱动参数缺少字段：{name}.{exc.args[0]}") from exc
    return tuple(fields)
