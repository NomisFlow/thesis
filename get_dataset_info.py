import os
import json
import numpy as np
from absl import flags
from absl import app

# Assuming this script is run from the project root (e.g., /Users/swolf/Desktop/thesis)
# Adjust paths if running from a different location.

# This path is hardcoded from the tmd-release/ogbench/utils.py file
DEFAULT_DATASET_DIR = '/Users/swolf/.gemini/tmp/d92b96257770d7f3e522e2c7c3fb5d0a4bf8fa3e595d79356f8fac67a38914f7/ogbench_data'

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, 'Environment name to inspect.')

def get_env_names_from_experiments(exp_dir='tmd-release/exp/OGBench'):
    """
    Reads flags.json from experiment directories to find used env_names.
    """
    env_names = set()
    if not os.path.exists(exp_dir):
        print(f"Experiment directory not found: {exp_dir}")
        return []

    for run_group in os.listdir(exp_dir):
        run_group_path = os.path.join(exp_dir, run_group)
        if os.path.isdir(run_group_path):
            for experiment_run in os.listdir(run_group_path):
                experiment_run_path = os.path.join(run_group_path, experiment_run)
                if os.path.isdir(experiment_run_path):
                    flags_file = os.path.join(experiment_run_path, 'flags.json')
                    if os.path.exists(flags_file):
                        try:
                            with open(flags_file, 'r') as f:
                                flags_data = json.load(f)
                                if 'env_name' in flags_data:
                                    env_names.add(flags_data['env_name'])
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON from {flags_file}")
    return sorted(list(env_names))

def get_dataset_size(env_name, dataset_dir=DEFAULT_DATASET_DIR):
    """
    Loads a dataset and returns the number of observations (transitions).
    """
    train_dataset_path = os.path.join(dataset_dir, f'{env_name}.npz')
    val_dataset_path = os.path.join(dataset_dir, f'{env_name}-val.npz')

    if not os.path.exists(train_dataset_path):
        print(f"Training dataset not found for {env_name} at {train_dataset_path}")
        return None, None

    try:
        with np.load(train_dataset_path) as data:
            train_size = len(data['observations'])
        print(f"Train dataset for '{env_name}': {train_size} transitions")
    except Exception as e:
        print(f"Error loading train dataset for {env_name}: {e}")
        train_size = None

    if os.path.exists(val_dataset_path):
        try:
            with np.load(val_dataset_path) as data:
                val_size = len(data['observations'])
            print(f"Validation dataset for '{env_name}': {val_size} transitions")
        except Exception as e:
            print(f"Error loading validation dataset for {env_name}: {e}")
            val_size = None
    else:
        print(f"Validation dataset not found for {env_name} at {val_dataset_path}")
        val_size = None

    return train_size, val_size

def main(_):
    if FLAGS.env_name:
        print(f"Attempting to get size for environment: {FLAGS.env_name}")
        get_dataset_size(FLAGS.env_name)
    else:
        print("No --env_name specified. Listing available environment names from experiment logs:")
        env_names = get_env_names_from_experiments()
        if env_names:
            print("\nFound the following environment names in experiment logs:")
            for name in env_names:
                print(f"- {name}")
            print(f"\nTo get the size of a specific dataset, run this script with --env_name <name>.")
            print(f"Example: python {__file__} --env_name {env_names[0]}")
        else:
            print("No environment names found in experiment logs.")

if __name__ == '__main__':
    app.run(main)
