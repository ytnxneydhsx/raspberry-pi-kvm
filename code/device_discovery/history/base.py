"""设备连接历史仓库的抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import DeviceHistoryRecord


class DeviceHistoryStore(ABC):
    """编排层依赖的设备连接历史保存接口。"""

    @abstractmethod
    def initialize(self) -> None:
        """准备历史目录和文件；已有内容必须保留。"""
        raise NotImplementedError

    @abstractmethod
    def append(self, record: DeviceHistoryRecord) -> None:
        """追加一条完整记录，持久化成功后返回。"""
        raise NotImplementedError
