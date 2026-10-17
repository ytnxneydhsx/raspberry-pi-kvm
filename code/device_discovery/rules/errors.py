"""规则模块对外公开的异常类型。"""


class RuleError(Exception):
    """规则模块错误的公共基类。"""


class RuleConfigurationError(RuleError):
    """规则模块的 YAML 配置缺失或无效。"""


class InvalidRuleError(RuleError):
    """调用者提供的 Key 或 VID/PID 不合法。"""


class RuleConflictError(RuleError):
    """Key 或完整 VID/PID 二元组已被占用。"""


class RuleNotFoundError(RuleError):
    """修改或删除的 Key 不存在。"""


class RuleFileError(RuleError):
    """规则文件、锁文件或保存操作发生错误。"""


class RuleFileFormatError(RuleFileError):
    """已有规则文件格式错误或违反唯一性约束。"""
