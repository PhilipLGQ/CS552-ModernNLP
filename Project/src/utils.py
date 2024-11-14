import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        response_str = output.split(self.template["response_split"])[1].strip()
        if response_str.endswith("</s>"):
            response_str = response_str[:-len("</s>")].strip()
        return response_str
    

def process_prompt_data(prompts_r_path: str, prompts_w_path: str):
    def filter_sample_with_explanation(prompts):
        samples_with_explanation = []
        samples_without_explanation = []
        for prompt in prompts:
            if 'explanation' in prompt.keys() and prompt['explanation'] is not None:
                samples_with_explanation.append(prompt)
            else:
                samples_without_explanation.append(prompt)
        return samples_with_explanation, samples_without_explanation

    def process_explanation(prompts, expl_flag=False):
        if expl_flag:
            for prompt in prompts:
                if isinstance(prompt['answer'], list) and len(prompt['answer']) > 1:  # if answers are multiples
                    assistant_response = "Correct answers are: " + ", ".join(prompt['answer'])
                else:
                    assistant_response = "Correct answer is: " + str(prompt['answer'])

                user_query = prompt['question'] + "\nchoices:\n" + '\n'.join(prompt['choices'])
                prompt['question'] = user_query
                prompt['answer'] = f"{assistant_response}, Explanation: {prompt['explanation']}"
        else:
            for prompt in prompts:
                if isinstance(prompt['answer'], list) and len(prompt['answer']) > 1:  # if answers are multiples
                    assistant_response = "Correct answers are: " + ", ".join(prompt['answer'])
                else:
                    assistant_response = "Correct answer is: " + str(prompt['answer'])

                user_query = prompt['question'] + "\nchoices:\n" + '\n'.join(prompt['choices'])
                prompt['question'] = user_query
                prompt['answer'] = f"{assistant_response}"
        return prompts

    with open(prompts_r_path, "r") as f:
        samples = json.load(f)

    total_length = len(samples)
    empty_count = 0

    for sample in samples:
        if isinstance(sample['answer'], list):
            if not sample['answer']:
                samples.remove(sample)
                empty_count += 1
    print(f"Inspected {empty_count} samples with empty answer. Remaining {len(samples)} out of {total_length}.")

    TFs, MCQs, SAQs, what_else = [], [], [], []
    for sample in samples:
        # Check if the sample contains valid choices
        if 'choices' in sample.keys() and sample['choices'] is not None:
            if all(choice.lower() in ['true', 'false'] for choice in sample['choices']):
                TFs.append(sample)  # check if these choices are True or False
            else:
                MCQs.append(sample)  # check if these choices are arbitrary choices of MCQs
        elif ('choices' not in sample.keys() or sample['choices'] is None) and (
                'answer' in sample.keys() and sample['answer'] is not None):
            SAQs.append(sample)
        else:  # if the sample contains neither choices nor answer
            if set(sample.keys()) == {'question', 'sol_id'}:
                what_else.append(sample)

    assert len(MCQs) + len(TFs) + len(SAQs) + len(what_else) == len(
        samples), "Sample(s) with wrong form exist(s). Please check the original dataset..."

    # MCQ
    MCQs_with_explanation, MCQs_without_explanation = filter_sample_with_explanation(MCQs)
    if len(MCQs_without_explanation) > 0:
        MCQs_with_explanation = process_explanation(MCQs_with_explanation, expl_flag=True)
    if len(MCQs_without_explanation) > 0:
        MCQs_without_explanation = process_explanation(MCQs_without_explanation, expl_flag=False)
    MCQs_all = MCQs_with_explanation + MCQs_without_explanation

    # TF
    TFs_with_explanation, TFs_without_explanation = filter_sample_with_explanation(TFs)
    if len(TFs_with_explanation) > 0:
        TFs_with_explanation = process_explanation(TFs_with_explanation, expl_flag=True)
    if len(TFs_without_explanation) > 0:
        TFs_without_explanation = process_explanation(TFs_without_explanation, expl_flag=False)
    TFs_all = TFs_with_explanation + TFs_without_explanation

    # SAQ
    for sample in SAQs:
        if isinstance(sample['answer'], list):
            if len(sample['answer']) > 1:
                sample['answer'] = ' '.join(sample['answer'])
            else:
                sample['answer'] = sample['answer'][0]

    # Save
    samples_all = MCQs_all + TFs_all + SAQs
    for sample in samples_all:
        if "explanation" in sample.keys():
            sample.pop("explanation")
        if "choices" in sample.keys():  
            sample.pop("choices")

    with open(prompts_w_path, 'w') as file:
        json.dump(samples_all, file, indent=4)
