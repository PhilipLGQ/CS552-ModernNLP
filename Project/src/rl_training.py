from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    LlamaTokenizer,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default="lora-alpaca-adapter_merged", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="yahma/llama-7b-hf", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="ethical_frugal_reward_model-adapter_merged", metadata={"help": "the reward model name"})
    # reward_model_name: Optional[str] = field(default="models/ethical-frugal-reward-model", metadata={"help": "the reward model name"})
    lora_weights: Optional[str] = field(default="lora-alpaca", metadata={"help": "the fine-tuned generative model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=True, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="models/ppo-fine-tuned-ethical-frugal-llama-500-epoch1", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    load_8bit: Optional[bool] = field(default=False, metadata={"help": "load 8bit model"})
    data_size: Optional[int] = field(default=3500, metadata={"help": "size of the training dataset"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)
train_dataset = load_dataset('json', data_files='dataset/NLI_filtered_data.json')
train_dataset = train_dataset['train'].select(range(script_args.data_size))  # change this number to use dataset of different size

reward_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

tokenizer = LlamaTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

def build_dataset(
    tokenizer,
    data_files="dataset/NLI_filtered_data.json",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load datasets
    ds = load_dataset('json', data_files=data_files)
    original_columns = ds["train"].column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# build the model
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=8, #16,
    lora_alpha=16, #32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.model_name,
    load_in_8bit=script_args.load_8bit,
    device_map={"": current_device},
    peft_config=lora_config,
    torch_dtype=torch.float16,
)

# use with pipeline
reward_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.reward_model_name, num_labels=1, torch_dtype=torch.bfloat16
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# build the reward computation pipeline
device = ppo_trainer.accelerator.device
# device = "cpu"
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug


qa_pipeline = pipeline(
    "sentiment-analysis",
    model=reward_model, # the reward model needs to be compatible
    device=device,
    # model_kwargs={"load_in_8bit": False},
    tokenizer=tokenizer,
    # return_token_type_ids=False,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break
    question_tensors = batch["input_ids"]

    #### Get response from the generative model
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    #### Compute rewards
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = qa_pipeline(texts, **reward_kwargs)
    rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    # encoded_texts = [reward_model.tokenizer(text, return_tensors="pt") for text in texts]
    # rewards = [reward_model.forward(encoded_text) for encoded_text in encoded_texts]

    #### Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

# save the model
# https://stackoverflow.com/questions/74593644/how-to-fix-no-token-found-error-while-downloading-hugging-face
ppo_trainer.save_pretrained(script_args.output_dir)
