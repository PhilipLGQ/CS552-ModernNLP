import json
import os
import sys
import warnings

import fire
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils import Prompter

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    max_generate_length: int = 256,  # The maximum number of new tokens to generate for response
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    prompt_dataset_dir: str = None, # The directory to the file storing the questions to prompt.
    prompt_save_dir: str = None, # The directory to save the generated texts.
    prompt_save_name: str = None, # File name of the JSON formed prompt results to save.
    convert_to_submission: bool = False, # Convert to submission form if required.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained("yahma/llama-7b-hf")
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            # torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=2.0,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        output_str = prompter.get_response(output)
        return output_str
    
    if prompt_dataset_dir is not None and prompt_save_dir is not None:
        if prompt_dataset_dir.endswith(".json") or prompt_dataset_dir.endswith(".jsonl"):
            data = load_dataset("json", data_files=prompt_dataset_dir)
        else:
            data = load_dataset(prompt_dataset_dir)

        guid_flag = False
        data['train'] = data['train'].rename_column("question", "instruction")
        if "answer" in data['train'].column_names:
            data['train'] = data['train'].remove_columns("answer")
        if "guid" in data['train'].column_names:
            guid_flag = True
        assert convert_to_submission and guid_flag, "The processed original data is not in the form expected."
        print(f"# Samples in prompting set: {len(data['train'])}\n")

        responses = []
        prompt_dataset = data['train']
        for _, prompt_item in tqdm(enumerate(prompt_dataset), total=len(prompt_dataset), desc="Test Generation"):
            response = evaluate(prompt_item['instruction'],
                                max_new_tokens=max_generate_length)
            if not convert_to_submission:
                if guid_flag:
                    output_dict = {
                        "guid": prompt_item['guid'],
                        "question": prompt_item['instruction'],
                        "answer": response
                    }
                else:
                    output_dict = {
                        "question": prompt_item['instruction'],
                        "answer": response
                    }
            else:
                output_dict = {
                  "guid": prompt_item['guid'],
                  "model_answer": response
                }
            responses.append(output_dict)
        
        if not os.path.exists(prompt_save_dir):
            os.makedirs(prompt_save_dir)
        if prompt_save_name is None:
            prompt_save_name = "prompt_result"
        with open(os.path.join(prompt_save_dir, prompt_save_name+".json"), "w") as f:
            json.dump(responses, f, indent=4)
        
        print(f"Successfully saved prompting result to {prompt_save_dir} in file '{prompt_save_name}.json'.")

    else:
        for instruction in [
            "Tell me about alpacas.",
            "Tell me about the president of Mexico in 2019.",
            "Tell me about the king of France in 2019.",
            "List all Canadian provinces in alphabetical order.",
            "Write a Python program that prints the first 10 Fibonacci numbers.",
            "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
            "Tell me five words that rhyme with 'shock'.",
            "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
            "Count up from 1 to 500.",
        ]:
            print("Instruction:", instruction)
            print("Response:", evaluate(instruction))
            print()


if __name__ == "__main__":
    fire.Fire(main)
