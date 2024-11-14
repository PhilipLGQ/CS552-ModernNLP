import torch
import torch.nn.init as init
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    DebertaV2Config,
    PreTrainedModel,
)

from peft import PeftModel


# ========================================================
# Below is an example of how you can implement your own 
# custom HuggingFace model and use it in the evaluation.
# 
# This is where you should implement your model and the
# get_rewards() function. 
# 
# If you want to extend an existing
# HuggingFace model, you can also add additional layers or 
# heads in this class.
# ========================================================


class CustomRewardModelConfig(DebertaV2Config):
    """
    This is an example config class for a custom HuggingFace model.
    
    - It currently inherits from the DebertaV2Config class,
    because we are using the OpenAssistant Dberta model as our base model.
    
    - You are not expected to follow this example, but you can use it as a reference point.
    Inherit from the HuggingFace config class that is most similar to your base model.
    
    - Or, if you prefer, construct your own config class from scratch if you
    implement your base model from scratch.

    - You should specify the model_type as your model's class name.
    When loading the 
    """
    model_type = "CustomRewardModel"

    # If you have additional parameters to the model class,
    # you can add them inside the config class as well.
    # For example, with "def __init__(self, config, reward_dim=1):",
    # you can specify "reward_dim = 1" here in the config class.
    # Then, you can acess the reward_dim parameter in the model class 
    # by calling "self.config.reward_dim".

    def __init__(self, *args, adapter_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_path = adapter_path


class CustomRewardModel(PreTrainedModel):
    """
    This is an example regression model built on top of the OpenAssistant Dberta model.
    You are not expected to follow this example, but you can use it as a reference point.
    You should have the freedom to construct your model however you want.
    
    !IMPORTANT!: You need to implement the get_rewards() function, which takes in a list of demonstrations
                and returns a list of rewards. See more details in the fuction below.
    !IMPORTANT!: You should implement your model class such that 
                it can be saved as a HuggingFace PreTrainedModel.
                This menas you also need to implement the CustomHFConfig class 
                and specify the model_type as your model's class name.
    """

    # # Set the config class to your custom config class
    config_class = CustomRewardModelConfig

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass

    def __init__(self, config):
        super().__init__(config)

        # Initialize the base model and its associated tokenizer
        hf_pretrained_model_name = "Alvor/reward-model-deberta-v3-base-MIC-ethical-epfl-cs552"

        self.tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_pretrained_model_name)

        if config.adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                config.adapter_path,
                device_map={"": "mps"},  # Hardcoded since I only run locally
            )

        # Initialize the reward head if you want to add additional layers
        self.reward_head = torch.nn.Linear(1, 1)
        init.xavier_normal_(self.reward_head.weight)

        # Set the config
        self.config = config

    def forward(self, encoded):
        outputs = self.model(**encoded)
        logits = outputs.logits

        # return logits through the reward head
        rewards = self.reward_head(logits)
        return rewards

    def get_rewards(self, demonstrations):
        """
        Get the rewards for the demonstrations
        TODO: This is an example function, replace this with your actual implementation!
              Your implementation should handle the input and output format as specified below.
        
        Args:
            demonstrations: list of dicts in the format of
            {'chosen': str, 'rejected': str}
        Return:
            rewards: list of dicts in the format of
            {'chosen': float, 'rejected': float} 
        """
        rewards = []
        for pair in demonstrations:
            encoded_chosen = self.tokenizer(
                pair['chosen'], return_tensors="pt")
            encoded_reject = self.tokenizer(
                pair['rejected'], return_tensors="pt")
            scores_chosen = self.forward(encoded_chosen)
            scores_reject = self.forward(encoded_reject)
            rewards.append({
                'chosen': scores_chosen.item(),
                'rejected': scores_reject.item()
            })
        return rewards
