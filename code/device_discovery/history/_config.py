"""从 application.yaml 读取设备连接历史配置。"""

from __future__ import annotations

from pathlib import Path

from .errors import HistoryConfigurationError

_CODE_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE = _CODE_ROOT / "config" / "application.yaml"


def load_history_storage_file() -> Path:
    """读取 device_discovery.history.storage_file 并返回绝对路径。"""
    try:
        import yaml
    except ImportError as exc:
        raise HistoryConfigurationError(
            "缺少 PyYAML，请先安装 code/requirements.txt 中的依赖"
        ) from exc

    try:
        with _CONFIG_FILE.open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise HistoryConfigurationError(f"无法读取配置文件：{_CONFIG_FILE}") from exc

    if not isinstance(config, dict):
        raise HistoryConfigurationError("application.yaml 最外层必须是映射")

    device_discovery = config.get("device_discovery")
    if not isinstance(device_discovery, dict):
        raise HistoryConfigurationError("缺少配置段：device_discovery")

    history = device_discovery.get("history")
    if not isinstance(history, dict):
        raise HistoryConfigurationError("缺少配置段：device_discovery.history")

    storage_file = history.get("storage_file")
    if not isinstance(storage_file, str) or not storage_file.strip():
        raise HistoryConfigurationError(
            "device_discovery.history.storage_file 必须是非空字符串"
        )

    path = Path(storage_file).expanduser()
    if not path.is_absolute():
        path = _CONFIG_FILE.parent / path
    return path.resolve()
