import yaml

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
