"""规则模块对外使用的数据结构。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """规则快照；规则仓库返回的实例已经过校验和规范化。"""

    key: str
    vid: str
    pid: str
