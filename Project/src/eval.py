import os
import json
import numpy as np
import torch
from utils import process_prompt_data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluate import load
import argparse


class TextEvaluator:
    """
    This class is used to evaluate the quality of the generated text.
    """

    def __init__(self, root_path, batch_size=16):
        self.root_path = root_path
        self.batch_size = batch_size

        # Load bertscore model, as well as bart-large-mnli for NLI
        self.bertscore = load("bertscore")
        self.nli_score_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.nli_score_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

        # Determine the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nli_score_model = self.nli_score_model.to(self.device)

        # Variables to store results
        self.guid_list = []
        self.pred_list = []
        self.gt_list = []
        self.bertscore_result = None
        self.nli_all_probs = []
        self.nli_results = {}

    def load_data(self, ground_truth_file, prediction_file):
        self.prediction_file = prediction_file

        with open(ground_truth_file, 'r') as file:
            self.ground_truth_data = json.load(file)
          
        with open(prediction_file, 'r') as file:
            self.prediction_data = json.load(file)

    def compute_bert_score(self, lang="fr"):
        self.guid_list = []
        self.pred_list = []
        self.gt_list = []

        for sample_pred in self.prediction_data:
            for sample_gt in self.ground_truth_data:
                if sample_gt['guid'] == sample_pred['guid']:
                    self.guid_list.append(sample_gt['guid'])
                    self.pred_list.append(sample_pred['model_answer'])
                    self.gt_list.append(sample_gt['answer'])

        # Compute BERTScore, Notice: lang=fr will automatically download multi-lingual bart based model
        self.bertscore_result = self.bertscore.compute(predictions=self.pred_list,
                                                       references=self.gt_list,
                                                       lang=lang)

    def evaluate_nli_class(self):
        premises_hypotheses = []

        for sample_pred in self.prediction_data:
            for sample_gt in self.ground_truth_data:
                if sample_gt['guid'] == sample_pred['guid']:
                    premise = sample_gt['answer']
                    hypothesis = sample_pred['model_answer']
                    sequence = premise + ' ' + hypothesis
                    premises_hypotheses.append(sequence)

        # Split data into batches, especially useful for evaluating long sequence, even if you are using A100
        chunks = [premises_hypotheses[i:i + self.batch_size] for i in
                  range(0, len(premises_hypotheses), self.batch_size)]

        self.nli_all_probs = []

        for chunk in tqdm(chunks):
            inputs = self.nli_score_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)

            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.nli_score_model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()
            self.nli_all_probs.extend(probs)

        # For each sample in the batch
        for i, sample_probs in enumerate(self.nli_all_probs):
            guid = self.guid_list[i]
            contradiction_prob, neutral_prob, entailment_prob = sample_probs

            # Determine the class based on the probabilities
            if contradiction_prob > neutral_prob and contradiction_prob > entailment_prob:
                nli_class = "contradiction"
            elif entailment_prob > neutral_prob and entailment_prob > contradiction_prob:
                nli_class = "entailment"
            else:
                # Neutral is the highest value
                if contradiction_prob > entailment_prob:
                    nli_class = "weak contradiction"
                else:
                    nli_class = "weak entailment"

            self.nli_results[guid] = nli_class

        self.nli_all_probs = np.array(self.nli_all_probs)

    def summary_result(self):
        # Compute average BERTScore
        precision_avg = np.mean(self.bertscore_result['precision'])
        recall_avg = np.mean(self.bertscore_result['recall'])
        f1_avg = np.mean(self.bertscore_result['f1'])

        # Count the occurrences of each NLI class
        nli_class_counts = {
            "entailment": 0,
            "weak entailment": 0,
            "weak contradiction": 0,
            "contradiction": 0,
        }

        for nli_class in self.nli_results.values():
            nli_class_counts[nli_class] += 1

        # Calculate accuracy as (entailment + weak entailment) / all
        nli_accuracy = (nli_class_counts["entailment"] + nli_class_counts["weak entailment"]) / len(self.nli_results)

        # Print summary
        border = '-' * 80

        # Prepare NLI class counts string with more indentation and bar on the right-hand side
        nli_class_counts_str = '\n'.join(
            [f"|   {key}: {value}".ljust(78) + " |" for key, value in nli_class_counts.items()])

        # Print summary with borders
        print('\n\n')
        print(border)
        print("|" + " " * 78 + "|")
        print(f"| Average Precision: {precision_avg:.4f}".ljust(78) + " |")
        print(f"| Average Recall: {recall_avg:.4f}".ljust(78) + " |")
        print(f"| Average F1: {f1_avg:.4f}".ljust(78) + " |")
        print("| NLI Class Counts:".ljust(78) + " |")
        print(nli_class_counts_str)
        print(f"| NLI Accuracy (entailment + weak entailment / all): {nli_accuracy:.4f}".ljust(78) + " |")
        print("|" + " " * 78 + "|")
        print(border)

    def save_result(self, output_filename_suffix="_evaluated.json"):
        # Copy the original prediction data to avoid modifying it
        evaluated_data = self.prediction_data.copy()

        # Create a dictionary with GUID as key for easy lookup
        guid_to_index = {guid: index for index, guid in enumerate(self.guid_list)}

        # Iterate through each QA pair and store the corresponding results
        for qa_pair in evaluated_data:
            guid = qa_pair['guid']
            index = guid_to_index.get(guid)

            # Check if the index exists, if not it means the guid was not found in the evaluation results
            if index is not None:
                qa_pair['precision'] = float(self.bertscore_result['precision'][index])
                qa_pair['recall'] = float(self.bertscore_result['recall'][index])
                qa_pair['f1'] = float(self.bertscore_result['f1'][index])
                qa_pair['nli_class'] = self.nli_results.get(guid, "N/A")
            else:
                qa_pair['precision'] = "N/A"
                qa_pair['recall'] = "N/A"
                qa_pair['f1'] = "N/A"
                qa_pair['nli_class'] = "N/A"

        # Generate the output filename by appending the suffix
        output_filename = os.path.splitext(self.prediction_file)[0] + output_filename_suffix

        # Save the results to a new JSON file
        with open(output_filename, 'w') as file:
            json.dump(evaluated_data, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Evaluate text using BERTScore and NLI.')

    # Arguments
    parser.add_argument('--root-path', type=str, required=True, help='The project root directory.')
    parser.add_argument('--batch-size', type=str, default=16, required=True,
                        help='Batch size for calculating BERTScore.')
    parser.add_argument('--input-prompt-file', type=str, required=True, help='Path to the input prompt JSON file.')
    parser.add_argument('--model-prediction-file', type=str, required=True,
                        help='Relative path from root directory with filename of the model prediction JSON file.')

    args = parser.parse_args()

    # Process the ground truth prompt data
    base_name, file_extension = os.path.splitext(args.input_prompt_file)
    path_to_processed_gt_file = base_name + '_processed' + file_extension
    process_prompt_data(prompts_r_path=args.input_prompt_file, prompts_w_path=path_to_processed_gt_file)

    # Create TextEvaluator instance and load data
    evaluator = TextEvaluator(root_path=args.root_path)
    evaluator.load_data(ground_truth_file=path_to_processed_gt_file, prediction_file=args.model_prediction_file)

    # Compute BERTScore
    evaluator.compute_bert_score()

    # Evaluate NLI class
    evaluator.evaluate_nli_class()

    # Display summary result
    evaluator.summary_result()

    # Save results to file
    evaluator.save_result()


if __name__ == "__main__":
    main()