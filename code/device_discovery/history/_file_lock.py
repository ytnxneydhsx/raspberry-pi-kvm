"""设备连接历史文件的内部锁实现。"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import HistoryFileError


@contextmanager
def history_file_lock(lock_path: Path) -> Iterator[None]:
    """获取独占文件锁；锁随本次文件描述符关闭而释放。"""
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise HistoryFileError(f"无法打开历史锁文件：{lock_path}") from exc

    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise HistoryFileError(f"无法获取历史文件锁：{lock_path}") from exc
        yield
    finally:
        os.close(descriptor)
