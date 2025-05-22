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

    # Merge global and experiment configs, experiment taking precedence
    merged_config = {**global_config, **experiment_config}

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

if __name__ == '__main__':
    # Create a dummy train_config.yaml for testing
    dummy_yaml_content = """
global:
  model_name: UNet
  base_channels: 64
  dropout_rate: 0.1
  use_dilated_conv: false
  learning_rate: 0.001

experiment_A:
  base_channels: 128  # Overrides global
  dropout_rate: 0.15 # Overrides global
  batch_size: 32
  augmentation:
    hf_prob: 0.5
    cj_brightness: 0.2

experiment_B:
  use_dilated_conv: true # Overrides global
  learning_rate: 0.0005 # Overrides global
  batch_size: 16
  augmentation:
    hf_prob: 0.3
    cj_brightness: 0.1
    smart_crop:
      min_factor: 0.5
      max_factor: 0.9
"""
    with open('train_config_dummy.yaml', 'w') as f:
        f.write(dummy_yaml_content)

    print("Testing with 'experiment_A':")
    try:
        config_a = get_model_config('train_config_dummy.yaml', 'experiment_A')
        print(config_a)
    except Exception as e:
        print(e)

    print("\nTesting with 'experiment_B':")
    try:
        config_b = get_model_config('train_config_dummy.yaml', 'experiment_B')
        print(config_b)
    except Exception as e:
        print(e)

    print("\nTesting with a non-existent experiment 'experiment_C':")
    try:
        config_c = get_model_config('train_config_dummy.yaml', 'experiment_C')
        print(config_c)
    except Exception as e:
        print(e)

    print("\nTesting with a non-existent file:")
    try:
        config_d = get_model_config('non_existent_config.yaml', 'experiment_A')
        print(config_d)
    except Exception as e:
        print(e)
    
    # Clean up dummy file
    import os
    os.remove('train_config_dummy.yaml')
