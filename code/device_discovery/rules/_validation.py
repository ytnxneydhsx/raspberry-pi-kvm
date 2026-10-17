"""规则输入与文件内容的内部校验。"""

from __future__ import annotations

import re

from .errors import InvalidRuleError, RuleConflictError
from .models import Rule


def normalize_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise InvalidRuleError("Key 必须是非空字符串")
    return key.strip()


def normalize_usb_id(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidRuleError("VID/PID 必须是十六进制字符串")
    normalized = value.strip().lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    if re.fullmatch(r"[0-9a-f]{1,4}", normalized) is None:
        raise InvalidRuleError("VID/PID 必须包含 1 至 4 位十六进制数字")
    return normalized.zfill(4)


def build_rule(key: str, vid: str, pid: str) -> Rule:
    """校验输入并生成规范化规则。"""
    return Rule(
        key=normalize_key(key),
        vid=normalize_usb_id(vid),
        pid=normalize_usb_id(pid),
    )


def ensure_unique(rules: dict[str, Rule], candidate: Rule) -> None:
    """检查 Key 和完整 VID/PID 二元组的一一映射。"""
    if candidate.key in rules:
        raise RuleConflictError(f"Key 已存在：{candidate.key}")
    for rule in rules.values():
        if (rule.vid, rule.pid) == (candidate.vid, candidate.pid):
            raise RuleConflictError(
                f"VID/PID {candidate.vid}:{candidate.pid} 已属于 {rule.key}"
            )
