import copy
import yaml


def _deep_merge(dest: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dest`` and return ``dest``."""
    for key, value in src.items():
        if (
            key in dest
            and isinstance(dest[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(dest[key], value)
        else:
            dest[key] = copy.deepcopy(value)
    return dest

def get_model_config(yaml_path: str, experiment_name: str) -> dict:
    """
    Loads a YAML configuration file and merges global and experiment-specific configurations.

    Args:
        yaml_path: Path to the YAML configuration file.
        experiment_name: Name of the experiment to load configuration for.

    Returns:
        A dictionary containing the merged configuration.

    Raises:
        ValueError: If the experiment_name is not found in the YAML file.
        FileNotFoundError: If the yaml_path does not exist.
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Configuration file not found at {yaml_path}")

    global_config = config.get('global', {})
    experiment_config = config.get(experiment_name)

    if experiment_config is None:
        raise ValueError(f"Error: Experiment '{experiment_name}' not found in {yaml_path}")

    # Deep merge global and experiment configs, with experiment taking precedence
    merged_config = copy.deepcopy(global_config)
    _deep_merge(merged_config, experiment_config)

    return merged_config


def load_experiment_configs(yaml_path: str) -> list[dict]:
    """Load and merge experiment configurations from a YAML file.

    The YAML is expected to contain a ``global`` block and a list of
    experiment dictionaries under ``experiments``. Optional ``evaluation`` and
    ``logging`` blocks are merged into each experiment as well.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A list of dictionaries, one for each experiment, with all settings
        merged.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        ValueError: If an experiment entry does not contain a ``name``.
    """
    try:
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Configuration file not found at {yaml_path}")

    global_cfg = cfg.get("global", {})
    eval_cfg = cfg.get("evaluation", {})
    log_cfg = cfg.get("logging", {})
    experiments = cfg.get("experiments", [])

    if not isinstance(experiments, list):
        raise ValueError("'experiments' section must be a list")

    all_configs = []
    for exp in experiments:
        if "name" not in exp:
            raise ValueError("Experiment entry missing 'name'")

        merged: dict = {}
        _deep_merge(merged, global_cfg)
        if eval_cfg:
            merged.setdefault("evaluation", {})
            _deep_merge(merged["evaluation"], eval_cfg)
        if log_cfg:
            merged.setdefault("logging", {})
            _deep_merge(merged["logging"], log_cfg)
        _deep_merge(merged, exp)
        all_configs.append(merged)

    return all_configs