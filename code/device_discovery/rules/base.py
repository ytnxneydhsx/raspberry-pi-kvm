"""规则仓库的抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Rule


class RuleStore(ABC):
    """编排层依赖的规则仓库接口。"""

    @abstractmethod
    def initialize(self) -> None:
        """初始化存储；已有内容必须保留并通过校验。"""
        raise NotImplementedError

    @abstractmethod
    def list_rules(self) -> list[Rule]:
        """返回当前规则快照。"""
        raise NotImplementedError

    @abstractmethod
    def match(self, vid: str, pid: str) -> str | None:
        """按完整 VID/PID 返回 Key；未匹配返回 None。"""
        raise NotImplementedError

    @abstractmethod
    def add(self, key: str, vid: str, pid: str) -> Rule:
        """新增并保存规则。"""
        raise NotImplementedError

    @abstractmethod
    def update(
        self, key: str, vid: str, pid: str, *, new_key: str | None = None
    ) -> Rule:
        """更新并保存规则，可同时修改 Key。"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> Rule:
        """删除并返回原规则。"""
        raise NotImplementedError
