import unittest
import yaml
import os
import tempfile
from src.config_utils import get_model_config

class TestConfigUtils(unittest.TestCase):

    def setUp(self):
        # Create a temporary YAML file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.yaml_path = os.path.join(self.temp_dir.name, "test_config.yaml")

        self.test_yaml_content = {
            "global": {
                "model_name": "UNet",
                "base_channels": 64,
                "dropout_rate": 0.1,
                "learning_rate": 0.001
            },
            "experiment_A": {
                "base_channels": 128, # Overrides global
                "batch_size": 32,
                "augmentation": {
                    "hf_prob": 0.5
                }
            },
            "experiment_B": {
                "dropout_rate": 0.15, # Overrides global
                "learning_rate": 0.0005, # Overrides global
                "use_dilated_conv": True
            }
        }
        with open(self.yaml_path, 'w') as f:
            yaml.dump(self.test_yaml_content, f)

    def tearDown(self):
        # Clean up the temporary directory and file
        self.temp_dir.cleanup()

    def test_successful_merge_experiment_A(self):
        config = get_model_config(self.yaml_path, "experiment_A")
        
        # Test overridden value
        self.assertEqual(config.get("base_channels"), 128)
        # Test value from global
        self.assertEqual(config.get("learning_rate"), 0.001)
        # Test experiment-specific value
        self.assertEqual(config.get("batch_size"), 32)
        # Test nested experiment-specific value
        self.assertIn("augmentation", config)
        self.assertEqual(config["augmentation"].get("hf_prob"), 0.5)
        # Test global value not overridden
        self.assertEqual(config.get("model_name"), "UNet")

    def test_successful_merge_experiment_B(self):
        config = get_model_config(self.yaml_path, "experiment_B")

        # Test overridden values
        self.assertEqual(config.get("dropout_rate"), 0.15)
        self.assertEqual(config.get("learning_rate"), 0.0005)
        # Test experiment-specific value
        self.assertTrue(config.get("use_dilated_conv"))
        # Test global value not overridden
        self.assertEqual(config.get("base_channels"), 64)
        self.assertEqual(config.get("model_name"), "UNet")
        # Test value not present in this experiment but in global
        self.assertNotIn("batch_size", config) # batch_size is not in global or experiment_B

    def test_value_error_for_non_existent_experiment(self):
        with self.assertRaises(ValueError) as context:
            get_model_config(self.yaml_path, "experiment_C")
        self.assertTrue("Error: Experiment 'experiment_C' not found" in str(context.exception))

    def test_file_not_found_error(self):
        non_existent_path = os.path.join(self.temp_dir.name, "non_existent.yaml")
        with self.assertRaises(FileNotFoundError) as context:
            get_model_config(non_existent_path, "experiment_A")
        self.assertTrue(f"Error: Configuration file not found at {non_existent_path}" in str(context.exception))

    def test_empty_experiment_config(self):
        # Add an experiment with no specific overrides, should just get global
        self.test_yaml_content["experiment_empty"] = {}
        with open(self.yaml_path, 'w') as f:
            yaml.dump(self.test_yaml_content, f)
        
        config = get_model_config(self.yaml_path, "experiment_empty")
        self.assertEqual(config.get("model_name"), "UNet")
        self.assertEqual(config.get("base_channels"), 64)
        self.assertEqual(config.get("dropout_rate"), 0.1)
        self.assertEqual(config.get("learning_rate"), 0.001)

    def test_global_only_if_experiment_missing_fields(self):
        # Experiment A has base_channels, batch_size, augmentation
        # Global has model_name, base_channels, dropout_rate, learning_rate
        config = get_model_config(self.yaml_path, "experiment_A")
        expected_config = {
            "model_name": "UNet",       # from global
            "base_channels": 128,       # from experiment_A (overrides global)
            "dropout_rate": 0.1,        # from global
            "learning_rate": 0.001,     # from global
            "batch_size": 32,           # from experiment_A
            "augmentation": {           # from experiment_A
                "hf_prob": 0.5
            }
        }
        self.assertEqual(config, expected_config)

if __name__ == '__main__':
    # Create src directory if it doesn't exist, for config_utils import
    if not os.path.exists('src'):
        os.makedirs('src')
    # Ensure __init__.py exists in src for imports to work
    if not os.path.exists('src/__init__.py'):
        with open('src/__init__.py', 'w') as f:
            pass # Create empty __init__.py
            
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
