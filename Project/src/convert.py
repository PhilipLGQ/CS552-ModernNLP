import json
import os

import fire

from utils import process_prompt_data


def convert_to_submission(previous_path: str, 
                          save_path: str,
                          initial_conversion: bool = False,  # If you're running the testing from the very beginning, turn on in shell
                         ):
    
    save_dir, _ = os.path.split(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not initial_conversion:
        print("\nConverting old version of prompting results into submission-required form.")
        revised_samples = []
        with open(previous_path, 'r') as f:
            samples = json.load(f)

        for sample in samples:
            if "guid" in sample.keys():
                revised_sample = {
                  "guid": sample["guid"],
                  "model_answer": sample["answer"]
                }
            else:
                revised_sample = sample
            revised_samples.append(revised_sample)

        with open(save_path, 'w') as g:
            json.dump(revised_samples, g, indent=4)
    else:
        print("\nConverting the original testing prompts file into standard evaluation form.")
        process_prompt_data(prompts_r_path=previous_path, 
                            prompts_w_path=save_path)
    print(f"Successfully saved the conversion result to {save_path}.")


if __name__ == "__main__":
    fire.Fire(convert_to_submission)
