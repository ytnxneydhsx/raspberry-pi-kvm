"""规则管理模块的公开入口。

业务代码只从这里导入。以下划线开头的模块属于内部实现。
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from ._config import load_rule_storage_file
from ._json_store import JsonFileRuleStore as _JsonFileRuleStore
from .base import RuleStore
from .errors import (
    InvalidRuleError,
    RuleConfigurationError,
    RuleConflictError,
    RuleError,
    RuleFileError,
    RuleFileFormatError,
    RuleNotFoundError,
)
from .models import Rule


_default_store: RuleStore | None = None
_default_store_lock = Lock()


def create_rule_store(path: str | Path | None = None) -> RuleStore:
    """创建新的 JSON 规则仓库；未传路径时读取 application.yaml。"""
    rule_file = load_rule_storage_file() if path is None else path
    return _JsonFileRuleStore(rule_file)


def get_rule_store() -> RuleStore:
    """取得进程内共享的默认规则仓库，并在首次取得时初始化存储。"""
    global _default_store

    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                store = create_rule_store()
                store.initialize()
                _default_store = store
    return _default_store


__all__ = [
    "InvalidRuleError",
    "Rule",
    "RuleConfigurationError",
    "RuleConflictError",
    "RuleError",
    "RuleFileError",
    "RuleFileFormatError",
    "RuleNotFoundError",
    "RuleStore",
    "create_rule_store",
    "get_rule_store",
]
