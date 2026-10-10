"""Read YAML training hyperparameters without constructing command interfaces."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(
    os.environ.get(
        "CRLO_CONFIG_DIR",
        PROJECT_ROOT / "config"
        if (PROJECT_ROOT / "config").is_dir()
        else Path(sys.prefix) / "share/crlo/config",
    )
)


class ConfigMap(dict):
    """Parameter mapping with attribute access for experiment implementations."""

    def __getattr__(self, name: str) -> Any:
        """Read a supplied parameter by name."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class StrictLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently overwriting parameters."""


def _mapping(loader: StrictLoader, node: yaml.MappingNode) -> ConfigMap:
    result = ConfigMap()
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError(f"Config keys must be strings: {key!r}")
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path: str | Path) -> ConfigMap:
    """Read the selected YAML file; paths are explicit and no defaults are merged."""
    path = Path(path)
    if path.suffix != ".yaml":
        raise ValueError("Configuration files must use the .yaml extension")
    with path.open() as stream:
        data = yaml.load(stream, Loader=StrictLoader)
    if not isinstance(data, ConfigMap):
        raise TypeError("Configuration must be a YAML mapping")
    return data
