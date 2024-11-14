#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2023-04-19 00:13:26
LastEditTime: 2023-04-19 00:16:00
LastEditors: Kun
Description: 
FilePath: /Alpaca-RLHF-PyTorch/merge_peft_adapter.py
'''

from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable
    model_name: Optional[str] = field(default="OpenAssistant/reward-model-deberta-v3-base", metadata={"help": "the model name"})
    adapter_name: Optional[str] = field(default="ethical_pretrained_base_model", metadata={"help": "the adapter name"})
    # base_model_name: Optional[str] = field(default="decapoda-research/llama-7b-hf", metadata={"help": "the model name"}) # my code: this is not used.
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print("script_args: ", script_args)

peft_model_id = script_args.adapter_name
peft_config = PeftConfig.from_pretrained(peft_model_id)
print("peft_config: ", peft_config)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
)
# model = AutoModelForSequenceClassification.from_pretrained(
#     script_args.model_name,
#     num_labels=1,
#     torch_dtype=torch.bfloat16,
#     # ValueError: Loading THUDM/chatglm-6b requires you to execute the configuration file in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.
#     # trust_remote_code=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
# using above code, it will raise exception "ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported."
# reference  https://github.com/huggingface/transformers/issues/22222
# Hi @candowu, thanks for raising this issue. This is arising, because the tokenizer in the config on the hub points to LLaMATokenizer. However, the tokenizer in the library is LlamaTokenizer.
# This is likely due to the configuration files being created before the final PR was merged in.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

model = model.merge_and_unload()

if script_args.output_name is None:
    output_name = f"{script_args.adapter_name}-adapter_merged"
    model.save_pretrained(output_name)
else:
    model.save_pretrained(f"{script_args.output_name}")