"""JSON-backed serial profile storage used by the device-access layer."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Iterator

from .errors import ProfileStoreError
from .models import SerialConfig


_VERSION = 1
_PROFILE_FIELDS = tuple(field.name for field in fields(SerialConfig))


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中存在重复字段：{key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON 中不允许非标准数字：{value}")


class JsonSerialProfileStore:
    """Persist named :class:`SerialConfig` values in a locked JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._lock_path = self._path.with_name(self._path.name + ".lock")

    def initialize(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProfileStoreError(f"无法创建串口配置目录：{self._path.parent}") from exc

        with self._file_lock(exclusive=True):
            if os.path.lexists(self._path):
                self._read_profiles()
            else:
                self._write_profiles({})

    def list_profiles(self) -> dict[str, SerialConfig]:
        with self._file_lock(exclusive=False):
            profiles = self._read_profiles()
        return dict(profiles)

    def get(self, name: str) -> SerialConfig:
        normalized_name = self._normalize_name(name)
        with self._file_lock(exclusive=False):
            profiles = self._read_profiles()
        try:
            return profiles[normalized_name]
        except KeyError as exc:
            raise ProfileStoreError(f"串口配置不存在：{normalized_name}") from exc

    def save(self, name: str, config: SerialConfig) -> None:
        normalized_name = self._normalize_name(name)
        if not isinstance(config, SerialConfig):
            raise ProfileStoreError("串口配置必须是 SerialConfig 对象")

        with self._file_lock(exclusive=True):
            profiles = self._read_profiles()
            profiles[normalized_name] = config
            self._write_profiles(profiles)

    def delete(self, name: str) -> None:
        normalized_name = self._normalize_name(name)
        with self._file_lock(exclusive=True):
            profiles = self._read_profiles()
            if normalized_name not in profiles:
                raise ProfileStoreError(f"串口配置不存在：{normalized_name}")
            del profiles[normalized_name]
            self._write_profiles(profiles)

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise ProfileStoreError("串口配置名称必须是字符串")
        normalized = name.strip()
        if not normalized:
            raise ProfileStoreError("串口配置名称不能为空")
        return normalized

    def _read_profiles(self) -> dict[str, SerialConfig]:
        """Read and validate the file; the caller must hold a file lock."""
        if not os.path.lexists(self._path):
            raise ProfileStoreError(f"串口配置尚未初始化：{self._path}")
        try:
            with self._path.open("r", encoding="utf-8") as source:
                data = json.load(
                    source,
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
        except (OSError, UnicodeError, ValueError) as exc:
            if isinstance(exc, ProfileStoreError):
                raise
            raise ProfileStoreError(f"串口配置文件无效：{self._path}: {exc}") from exc

        if not isinstance(data, dict) or set(data) != {"version", "profiles"}:
            raise ProfileStoreError("串口配置文件必须且只能包含 version、profiles 字段")
        if type(data["version"]) is not int or data["version"] != _VERSION:
            raise ProfileStoreError("串口配置文件 version 必须为 1")
        raw_profiles = data["profiles"]
        if not isinstance(raw_profiles, dict):
            raise ProfileStoreError("串口配置文件 profiles 必须是对象")

        profiles: dict[str, SerialConfig] = {}
        try:
            for raw_name, raw_config in raw_profiles.items():
                name = self._normalize_name(raw_name)
                if name in profiles:
                    raise ProfileStoreError(f"串口配置名称重复：{name}")
                if not isinstance(raw_config, dict) or set(raw_config) != set(
                    _PROFILE_FIELDS
                ):
                    raise ProfileStoreError(
                        f"串口配置 {name!r} 必须且只能包含：{', '.join(_PROFILE_FIELDS)}"
                    )
                profiles[name] = SerialConfig(**raw_config)
        except ProfileStoreError:
            raise
        except Exception as exc:
            raise ProfileStoreError(f"串口配置字段无效：{exc}") from exc
        return profiles

    def _write_profiles(self, profiles: dict[str, SerialConfig]) -> None:
        """Write a complete file and atomically replace the destination."""
        data = {
            "version": _VERSION,
            "profiles": {
                name: {field: getattr(config, field) for field in _PROFILE_FIELDS}
                for name, config in profiles.items()
            },
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as target:
                temporary = Path(target.name)
                os.chmod(target.fileno(), 0o600)
                json.dump(data, target, ensure_ascii=False, indent=2, allow_nan=False)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self._path)
        except (OSError, TypeError, ValueError) as exc:
            raise ProfileStoreError(f"无法保存串口配置：{self._path}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @contextmanager
    def _file_lock(self, *, exclusive: bool) -> Iterator[None]:
        try:
            descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            raise ProfileStoreError(f"无法打开串口配置锁：{self._lock_path}") from exc

        try:
            try:
                operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(descriptor, operation)
            except OSError as exc:
                raise ProfileStoreError(f"无法获取串口配置锁：{self._lock_path}") from exc
            yield
        finally:
            os.close(descriptor)
