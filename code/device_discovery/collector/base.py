"""系统设备采集器的抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from .models import DeviceObservation

DeviceEventCallback = Callable[[DeviceObservation], None]


class DeviceCollector(ABC):
    """编排层依赖的设备扫描与监听接口。"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """监听线程当前是否正在运行。"""
        raise NotImplementedError

    @abstractmethod
    def scan(self) -> list[DeviceObservation]:
        """返回当前连接的物理 USB 设备及其资源。"""
        raise NotImplementedError

    @abstractmethod
    def start(self, callback: DeviceEventCallback) -> None:
        """启动监听；在后台线程中把相关 udev 事件交给回调。"""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """停止监听；未运行时调用也视为成功。"""
        raise NotImplementedError
