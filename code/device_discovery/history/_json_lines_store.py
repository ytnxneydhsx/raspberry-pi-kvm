"""DeviceHistoryStore 的 JSON Lines 文件实现。"""

from __future__ import annotations

import os
from pathlib import Path

from ._file_lock import history_file_lock
from ._serialization import serialize_record
from .base import DeviceHistoryStore
from .errors import HistoryFileError
from .models import DeviceHistoryRecord


class JsonLinesDeviceHistoryStore(DeviceHistoryStore):
    """在独占锁内向本机 JSON Lines 文件追加设备记录。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._lock_path = self._path.with_name(self._path.name + ".lock")

    def initialize(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HistoryFileError(f"无法创建历史目录：{self._path.parent}") from exc

        with history_file_lock(self._lock_path):
            existed = self._path.exists()
            descriptor = self._open_history_file(create=True)
            try:
                if not existed:
                    os.fchmod(descriptor, 0o600)
                    os.fsync(descriptor)
            except OSError as exc:
                raise HistoryFileError(f"无法初始化历史文件：{self._path}") from exc
            finally:
                os.close(descriptor)

    def append(self, record: DeviceHistoryRecord) -> None:
        payload = serialize_record(record)
        with history_file_lock(self._lock_path):
            descriptor = self._open_history_file(create=False)
            try:
                self._write_all(descriptor, payload)
                os.fsync(descriptor)
            except OSError as exc:
                raise HistoryFileError(f"无法追加历史记录：{self._path}") from exc
            finally:
                os.close(descriptor)

    def _open_history_file(self, *, create: bool) -> int:
        flags = os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        if create:
            flags |= os.O_CREAT
        try:
            return os.open(self._path, flags, 0o600)
        except OSError as exc:
            if not create and not os.path.lexists(self._path):
                message = "历史文件尚未初始化"
            else:
                message = "无法打开历史文件"
            raise HistoryFileError(f"{message}：{self._path}") from exc

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("历史文件写入未取得进展")
            offset += written
