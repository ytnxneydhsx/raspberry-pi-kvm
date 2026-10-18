"""读取串口配置方案的存储位置。"""

from __future__ import annotations

from pathlib import Path

from .errors import ProfileStoreError

_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "application.yaml"


def load_profile_path(config_path: str | Path | None = None) -> Path:
    path = Path(config_path).expanduser().resolve() if config_path is not None else _CONFIG_FILE
    try:
        import yaml
    except ImportError as exc:
        raise ProfileStoreError("缺少 PyYAML，请安装 code/requirements.txt 中的依赖") from exc
    try:
        with path.open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProfileStoreError(f"无法读取配置文件：{path}") from exc
    try:
        storage_file = config["device_access"]["serial_profiles"]["storage_file"]
    except (KeyError, TypeError) as exc:
        raise ProfileStoreError("缺少配置：device_access.serial_profiles.storage_file") from exc
    if not isinstance(storage_file, str) or not storage_file.strip():
        raise ProfileStoreError("串口配置方案的 storage_file 必须是非空字符串")
    storage_path = Path(storage_file).expanduser()
    if not storage_path.is_absolute():
        storage_path = path.parent / storage_path
    return storage_path.resolve()
