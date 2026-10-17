"""读取系统设备采集模块的 YAML 配置。"""

from __future__ import annotations

from pathlib import Path

import yaml

from .errors import CollectorConfigurationError

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "application.yaml"


def load_resource_subsystems(config_path: str | Path | None = None) -> tuple[str, ...]:
    """读取需要归并到 USB 父设备的 Linux 子系统名称。"""
    path = (
        _DEFAULT_CONFIG_PATH
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    try:
        with path.open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CollectorConfigurationError(f"无法读取设备采集配置：{path}") from exc

    try:
        raw_subsystems = config["device_discovery"]["collector"][
            "resource_subsystems"
        ]
    except (KeyError, TypeError) as exc:
        raise CollectorConfigurationError(
            "配置缺少 device_discovery.collector.resource_subsystems"
        ) from exc

    if not isinstance(raw_subsystems, list) or not raw_subsystems:
        raise CollectorConfigurationError("resource_subsystems 必须是非空列表")

    normalized: list[str] = []
    for item in raw_subsystems:
        if not isinstance(item, str) or not item.strip():
            raise CollectorConfigurationError(
                "resource_subsystems 中的每一项都必须是非空字符串"
            )
        subsystem = item.strip()
        if subsystem not in normalized:
            normalized.append(subsystem)
    return tuple(normalized)
