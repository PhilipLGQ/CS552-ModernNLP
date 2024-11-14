import argparse
import json
import random

import numpy as np
import torch

# Set a fixed random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Here we load your CustomRewardModelConfig and CustomRewardModel classes, 
# so we have the implementation of your get_rewards function
# and load the weights from your saved model.
from model import CustomRewardModel, CustomRewardModelConfig


def load_json(filename):
    """Load json file"""
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
    return data


def save_dictlist_to_json(mydictlist, filename):
    """Save a list of dictionaries to json file"""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4)
    f.close()


class TestDataset(Dataset):
    """Simple dataset module for testing the reward model"""

    def __init__(self, test_ds):
        self.ds = test_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ix):
        return self.ds[ix]


class Reward(torch.nn.Module):
    """
    Wrapper class for the reward model, 
    which handles loading the model and tokenizers, 
    and the forward pass for final predictions
    """

    def __init__(self, model_path):
        super().__init__()

        # Load student-defined reward model and its associated config
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, config=self.config)

    def check_reward_type(self, rewards):
        return isinstance(rewards, list) and all(isinstance(r, dict) for r in rewards)

    def forward(self, demonstrations):
        """
        Get the reward predictions from student's reward model
        Args:
            demonstrations: list of dicts in the format of 
            {'chosen': str, 'rejected': str}
        Return:
            rewards: list of dicts in the format of
            {'chosen': float, 'rejected': float} 
        """
        # ===== Get the rewards from student's reward model =====
        # NOTE: You should implement the "get_rewards" method in your reward model
        rewards = self.model.get_rewards(demonstrations)

        # ===== Check the reward format =====
        assert self.check_reward_type(rewards), "The rewards must be a list of dicts"
        assert len(rewards) == len(demonstrations), "The number of rewards must match the number of demonstration pairs"
        return rewards


class Evaluator:
    def __init__(self, model_path, ds_test):
        # Load the model and dataset
        self.load_model(model_path)
        self.ds_test = ds_test
        self.dataset = TestDataset(ds_test)
        self.dataloader = DataLoader(
            self.dataset, batch_size=2, shuffle=False,
            collate_fn=lambda x: x)

    def load_model(self, model_path):
        """Load the reward model from the specified path"""
        self.model = Reward(model_path)

    def evaluate(self):
        """Evaluate the model on the test dataset"""
        rewards = []
        correct_indices = []  # Store the indices of correct predictions

        for i, batch in enumerate(tqdm(self.dataloader)):
            batch_rewards = self.model(batch)
            rewards.extend(batch_rewards)

            # Check for correct predictions within the batch
            for j, reward in enumerate(batch_rewards):
                # print(idx, reward['chosen'], reward['rejected'])
                if reward['chosen'] > reward['rejected']:
                    correct_indices.append(i * self.dataloader.batch_size + j)

        num_correct = len(correct_indices)
        acc = num_correct / len(self.ds_test)
        print(f"Evaluation Complete, Accuracy: {acc}")
        print(f"Indices of Correct Predictions: {correct_indices}")

        for idx, elem in enumerate(rewards):
            print(f"{idx}: {elem}")


def save_hf_model(hf_model, model_path):
    """Save the model and tokenizer to the specified path"""
    hf_model.save_pretrained(model_path)
    hf_model.config.save_pretrained(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ethical_frugal_reward_model",
        help="Path to the model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/reward_model_test_dataset.json",
        help="Path to the test dataset")
    args = parser.parse_args()

    # # NOTE: Example code to save your reward model

    # TODO: Uncomment the line below if adapter is NOT merged to base model
    model_config = CustomRewardModelConfig(adapter_path=args.model_path)

    # TODO: Uncomment the line below if adapter is NOT merged to base model
    # model_config = CustomRewardModelConfig()  # If adapter is merged to base model

    model = CustomRewardModel(model_config)
    save_hf_model(model, args.model_path) # this line overwrites config and pytorch_model

    # NOTE: Example code of how we will load your dataset
    reward_dataset = load_json(args.data_path)

    # NOTE: Example of how we will load your model

    # Here we need to register the custom reward model and its config
    # to the AutoModel and AutoConfig classes, so that we can load
    # the model using the AutoModel and AutoConfig classes
    AutoConfig.register('CustomRewardModel', CustomRewardModelConfig)
    AutoModel.register(CustomRewardModelConfig, CustomRewardModel)

    evaluator = Evaluator(args.model_path, reward_dataset)
    evaluator.evaluate()
