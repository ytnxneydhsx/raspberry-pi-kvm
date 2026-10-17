"""设备连接历史模块的稳定公开入口。"""

from __future__ import annotations

import threading
from pathlib import Path

from ._config import load_history_storage_file
from ._json_lines_store import JsonLinesDeviceHistoryStore
from .base import DeviceHistoryStore
from .errors import (
    DeviceHistoryError,
    HistoryConfigurationError,
    HistoryFileError,
    InvalidHistoryRecordError,
)
from .models import DeviceHistoryRecord, DeviceHistoryResource

_default_history_store: DeviceHistoryStore | None = None
_default_history_store_lock = threading.Lock()


def create_history_store(path: str | Path | None = None) -> DeviceHistoryStore:
    """创建历史仓库；路径为空时读取 application.yaml。"""
    storage_file = load_history_storage_file() if path is None else path
    return JsonLinesDeviceHistoryStore(storage_file)


def get_history_store() -> DeviceHistoryStore:
    """返回当前进程共享且已经初始化的默认历史仓库。"""
    global _default_history_store
    if _default_history_store is None:
        with _default_history_store_lock:
            if _default_history_store is None:
                store = create_history_store()
                store.initialize()
                _default_history_store = store
    return _default_history_store


__all__ = [
    "DeviceHistoryError",
    "DeviceHistoryRecord",
    "DeviceHistoryResource",
    "DeviceHistoryStore",
    "HistoryConfigurationError",
    "HistoryFileError",
    "InvalidHistoryRecordError",
    "create_history_store",
    "get_history_store",
]
