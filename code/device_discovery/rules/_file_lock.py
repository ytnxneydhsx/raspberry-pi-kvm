"""规则文件的内部锁实现。"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import RuleFileError


@contextmanager
def file_lock(lock_path: Path, *, exclusive: bool) -> Iterator[None]:
    """获取共享锁或独占锁；锁随本次文件描述符关闭而释放。"""
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise RuleFileError(f"无法打开锁文件：{lock_path}") from exc

    try:
        try:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(descriptor, operation)
        except OSError as exc:
            raise RuleFileError(f"无法获取规则文件锁：{lock_path}") from exc
        yield
    finally:
        os.close(descriptor)
