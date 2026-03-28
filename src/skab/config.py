import os
from typing import Any, Dict, Tuple

import yaml


DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def load_skab_config(config_path: str) -> Tuple[Dict[str, Any], str]:
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)

    for section in ['dataset', 'model', 'training', 'paths']:
        if section not in config:
            raise KeyError(f"Missing required configuration section: {section}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    paths = config.setdefault('paths', {})
    for key, value in list(paths.items()):
        if isinstance(value, str) and not os.path.isabs(value):
            paths[key] = os.path.normpath(os.path.join(project_root, value))

    return config, project_root
