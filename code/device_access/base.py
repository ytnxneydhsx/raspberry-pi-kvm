"""供应用调用的串口能力约定。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from device_discovery.collector import DeviceResource

from .models import ChannelState, SerialConfig


class SerialChannel(ABC):
    """一个串口资源的独立连接；不把网络或烧录强制套入字节流接口。"""

    @property
    @abstractmethod
    def resource(self) -> DeviceResource:
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self) -> SerialConfig:
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> ChannelState:
        raise NotImplementedError

    @property
    @abstractmethod
    def last_error(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def configure(self, config: SerialConfig) -> None:
        """仅在未打开时更换配置；调用 open() 后才应用到设备。"""
        raise NotImplementedError

    @abstractmethod
    def open(self) -> None:
        """由用户显式连接，重复打开报状态错误。"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """释放连接；可重复调用。"""
        raise NotImplementedError

    @abstractmethod
    def read(self, size: int = 4096) -> bytes:
        """返回原始字节；不解码、不按行拼接、不保存正文。"""
        raise NotImplementedError

    @abstractmethod
    def write(self, data: bytes) -> int:
        """返回实际写入长度；调用方处理部分写入，模块不自动重发。"""
        raise NotImplementedError
