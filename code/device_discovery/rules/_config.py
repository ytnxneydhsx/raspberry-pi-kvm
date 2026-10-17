"""从 application.yaml 读取规则模块配置。"""

from __future__ import annotations

from pathlib import Path

from .errors import RuleConfigurationError


_CODE_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE = _CODE_ROOT / "config" / "application.yaml"


def load_rule_storage_file() -> Path:
    """读取 device_discovery.rules.storage_file 并返回绝对路径。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuleConfigurationError(
            "缺少 PyYAML，请先安装 code/requirements.txt 中的依赖"
        ) from exc

    try:
        with _CONFIG_FILE.open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuleConfigurationError(f"无法读取配置文件：{_CONFIG_FILE}") from exc

    if not isinstance(config, dict):
        raise RuleConfigurationError("application.yaml 最外层必须是映射")

    device_discovery = config.get("device_discovery")
    if not isinstance(device_discovery, dict):
        raise RuleConfigurationError("缺少配置段：device_discovery")

    rules = device_discovery.get("rules")
    if not isinstance(rules, dict):
        raise RuleConfigurationError("缺少配置段：device_discovery.rules")

    storage_file = rules.get("storage_file")
    if not isinstance(storage_file, str) or not storage_file.strip():
        raise RuleConfigurationError(
            "device_discovery.rules.storage_file 必须是非空字符串"
        )

    path = Path(storage_file).expanduser()
    if not path.is_absolute():
        path = _CONFIG_FILE.parent / path
    return path.resolve()
