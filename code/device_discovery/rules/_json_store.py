"""RuleStore 的 JSON 文件实现。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ._file_lock import file_lock
from ._validation import build_rule, ensure_unique, normalize_key, normalize_usb_id
from .base import RuleStore
from .errors import (
    InvalidRuleError,
    RuleConflictError,
    RuleFileError,
    RuleFileFormatError,
    RuleNotFoundError,
)
from .models import Rule


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝 JSON 重复字段，防止解析时静默覆盖。"""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中存在重复字段：{key}")
        result[key] = value
    return result


class JsonFileRuleStore(RuleStore):
    """基于 JSON 文件的规则仓库内部实现，无长期内存缓存。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        # 原子替换会更换规则文件对象，因此锁使用独立、稳定的文件。
        self._lock_path = self._path.with_name(self._path.name + ".lock")

    def initialize(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuleFileError(f"无法创建规则目录：{self._path.parent}") from exc

        with file_lock(self._lock_path, exclusive=True):
            # lexists 能识别断开的符号链接，避免误当成新文件覆盖。
            if os.path.lexists(self._path):
                self._read_rules()
            else:
                self._save_rules({})

    def list_rules(self) -> list[Rule]:
        with file_lock(self._lock_path, exclusive=False):
            rules = self._read_rules()
        return list(rules.values())

    def match(self, vid: str, pid: str) -> str | None:
        pair = (normalize_usb_id(vid), normalize_usb_id(pid))
        for rule in self.list_rules():
            if (rule.vid, rule.pid) == pair:
                return rule.key
        return None

    def add(self, key: str, vid: str, pid: str) -> Rule:
        rule = build_rule(key, vid, pid)
        with file_lock(self._lock_path, exclusive=True):
            rules = self._read_rules()
            ensure_unique(rules, rule)
            rules[rule.key] = rule
            self._save_rules(rules)
        return rule

    def update(
        self, key: str, vid: str, pid: str, *, new_key: str | None = None
    ) -> Rule:
        current_key = normalize_key(key)
        rule = build_rule(current_key if new_key is None else new_key, vid, pid)
        with file_lock(self._lock_path, exclusive=True):
            rules = self._read_rules()
            if current_key not in rules:
                raise RuleNotFoundError(f"规则不存在：{current_key}")
            del rules[current_key]
            ensure_unique(rules, rule)
            rules[rule.key] = rule
            self._save_rules(rules)
        return rule

    def delete(self, key: str) -> Rule:
        normalized_key = normalize_key(key)
        with file_lock(self._lock_path, exclusive=True):
            rules = self._read_rules()
            if normalized_key not in rules:
                raise RuleNotFoundError(f"规则不存在：{normalized_key}")
            removed = rules.pop(normalized_key)
            self._save_rules(rules)
        return removed

    def _read_rules(self) -> dict[str, Rule]:
        """调用者必须持有共享锁或独占锁。"""
        try:
            with self._path.open("r", encoding="utf-8") as source:
                data = json.load(source, object_pairs_hook=_unique_json_object)
        except (OSError, UnicodeError) as exc:
            raise RuleFileError(f"无法读取规则文件：{self._path}") from exc
        except ValueError as exc:
            raise RuleFileFormatError(f"规则 JSON 无效：{exc}") from exc

        if not isinstance(data, dict):
            raise RuleFileFormatError("规则文件最外层必须是 JSON 对象")

        rules: dict[str, Rule] = {}
        try:
            for key, value in data.items():
                if not isinstance(value, dict) or set(value) != {"vid", "pid"}:
                    raise RuleFileFormatError(
                        f"规则 {key!r} 必须且只能包含 vid、pid 字段"
                    )
                rule = build_rule(key, value["vid"], value["pid"])
                ensure_unique(rules, rule)
                rules[rule.key] = rule
        except (InvalidRuleError, RuleConflictError) as exc:
            raise RuleFileFormatError(f"规则文件内容无效：{exc}") from exc
        return rules

    def _save_rules(self, rules: dict[str, Rule]) -> None:
        """调用者必须持独占锁；写完整临时文件后原子替换。"""
        data = {
            rule.key: {"vid": rule.vid, "pid": rule.pid}
            for rule in rules.values()
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
                json.dump(data, target, ensure_ascii=False, indent=2)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self._path)
        except OSError as exc:
            raise RuleFileError(f"无法保存规则文件：{self._path}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    # 清理失败不掩盖原始保存错误或已成功的替换结果。
                    pass
