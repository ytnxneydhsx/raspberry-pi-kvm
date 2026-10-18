"""串口配置、通用配置项描述和连接状态。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

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


def get_serial_config_fields() -> tuple[ConfigField, ...]:
    """提供通用选项，不打开设备或探测波特率。"""
    defaults = SerialConfig()
    return (
        ConfigField("baudrate", "integer", defaults.baudrate,
                    (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600),
                    True, "常用建议值；可填写正整数，实际支持和精度由驱动与设备决定"),
        ConfigField("bytesize", "integer", defaults.bytesize, (5, 6, 7, 8)),
        ConfigField("parity", "string", defaults.parity, ("N", "E", "O", "M", "S")),
        ConfigField("stopbits", "number", defaults.stopbits, (1, 1.5, 2)),
        ConfigField("flow_control", "string", defaults.flow_control,
                    ("none", "rtscts", "xonxoff")),
        ConfigField("timeout", "number", defaults.timeout, allow_custom=True,
                    description="读取超时秒数；0 表示非阻塞，超时返回空字节不等于掉线"),
        ConfigField("write_timeout", "number", defaults.write_timeout, allow_custom=True,
                    description="写入超时秒数；发生超时可能已经发送部分数据"),
        ConfigField("dtr", "boolean", defaults.dtr, (False, True),
                    description="打开时请求的 DTR 状态；系统或驱动可能产生瞬态变化"),
        ConfigField("rts", "boolean", defaults.rts, (False, True),
                    description="无硬件流控时的 RTS 请求状态；RTS/CTS 模式由驱动控制"),
    )
