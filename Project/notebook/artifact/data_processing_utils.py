from collections import Counter
from gpt_wrapper.chat import Chat
from tqdm import tqdm
from itertools import combinations
import numpy as np
import pandas as pd
from typing import Dict, List
import gpt_wrapper
import re
import json
import matplotlib.pyplot as plt

# Define all possible roles
POSSIBLE_ROLES = ['system', 'user', 'assistant']

# Please replace the string with your own allocated API key
gpt_wrapper.api_key = "389407ee-4fdf-4dae-a053-bbbf725f5503"


def count_role_freq(sample):
    # Count the frequency of each role and include a count of 0 for roles that don't exist
    counter = Counter({role: 0 for role in POSSIBLE_ROLES})
    for interaction in sample['interaction']:
        if interaction['role'] in POSSIBLE_ROLES:
            counter[interaction['role']] += 1

    return counter


def transform_data(samples):
    """Index samples by sol_id and interaction_id, and also count the roles separately.

    Args:
        samples (list): A list of samples, each containing a sol_id and an interaction_id.

    Returns:
        dict, dict: Two nested dictionaries where the first level of keys are sol_id values and the second level of keys are interaction_id values.
                    The first dictionary contains the samples and the second dictionary contains the role counts.
    """

    # Initialize the dictionaries
    indexed_samples = {}
    role_freq = {}

    for sample in samples:
        sol_id = sample['sol_id']
        interaction_id = sample['interaction_id']

        # If this sol_id is not already in the dictionary, add it
        if sol_id not in indexed_samples:
            indexed_samples[sol_id] = {}
            role_freq[sol_id] = {}

        # Add the sample and role count to the dictionaries, indexed by sol_id and interaction_id
        indexed_samples[sol_id][interaction_id] = sample
        role_freq[sol_id][interaction_id] = count_role_freq(sample)

    # Convert role_freq to a DataFrame to facilitate analysis
    role_freq_df = pd.DataFrame.from_dict({(i, j): role_freq[i][j]
                                           for i in role_freq.keys()
                                           for j in role_freq[i].keys()},
                                          orient='index')

    return indexed_samples, role_freq_df


def transform_solution(samples, qtype):
    """
    """

    # Initialize the dictionaries
    indexed_samples = {}
    role_freq = {}

    for sample in samples:
        sol_id = sample['sol_id']

        # If this sol_id is not already in the dictionary, add it
        if sol_id not in indexed_samples:
            indexed_samples[sol_id] = {}
            role_freq[sol_id] = {}

        # Add the sample and role count to the dictionaries, indexed by sol_id
        indexed_samples[sol_id]['interaction'] = sample['interaction']
        indexed_samples[sol_id]['qtype'] = qtype
        indexed_samples[sol_id]['label'] = sample['label']

        role_freq[sol_id] = count_role_freq(sample)

    # Convert role_freq to a DataFrame to facilitate analysis
    role_freq_df = pd.DataFrame(role_freq).T

    return indexed_samples, role_freq_df


def filter_sample_with_explanation(samples):
    """
    Used when transforming solution into training samples
    Add system prompt if there is explanation for the sample
    """
    samples_with_explanation = []
    samples_without_explanation = []

    for sample in samples:
        if 'explanation' in sample.keys() and sample['explanation'] is not None:
            samples_with_explanation.append(sample)
        else:
            samples_without_explanation.append(sample)

    return samples_with_explanation, samples_without_explanation


def remove_bad_samples(raw_demo: dict, role_freq: pd.DataFrame):
    """Remove bad interaction samples that lacks 'user' or/and 'assistant' role(s) in the interaction.

    Args:
        raw_demo (dict): Nested dictionary with top-level key of sol_id and bottom-level key of interaction_id.
        role_freq (pandas.DataFrame): A dataframe recording the # occurrences of each role in each sample's interaction.

    Returns:
        removed_demo (dict): Nested dictionary with top-level key of sol_id and bottom-level key of interaction_id, with bad
                             samples removed.
    """
    removed_demo = raw_demo.copy()

    # Index to remove: samples without 'assistant' or/and 'user' role in the interaction
    no_assistant_idx = role_freq[role_freq['assistant'] == 0].index.to_list()
    no_user_idx = role_freq[role_freq['user'] == 0].index.to_list()
    idx_to_remove = list(set(no_user_idx + no_assistant_idx))

    for sol_id, interaction_id in idx_to_remove:
        if sol_id in removed_demo and interaction_id in removed_demo[sol_id]:
            removed_demo[sol_id].pop(interaction_id)
            if not removed_demo[sol_id]:
                removed_demo.pop(sol_id)

    return removed_demo


# Concatenation of multiple interactions in serial order and specialize roles in interactions
def combine_interaction(interaction_list):
    """
    Concatenates interactions in the given list into a single string.

    Args:
        interaction_list (list): A list of dictionaries where each dictionary represents an interaction. 
                                 Each dictionary should contain 'role' and 'content' keys.

    Returns:
        combined_str (str): A single string representation of the combined interactions in the format 
                            "<role_{role}>: {content}".
    """
    combined_str = ""
    for interaction in interaction_list:
        content = interaction['content'].rstrip()
        combined_str += "<role_{}>".format(interaction['role']) + ": " + content + "\n\n"
    return combined_str


def simple_interaction_combination(demo):
    """Concatenation and combination of each sample's interaction into a string. Ignore other properties.

    Args:
        demo (dict): Nested dictionary with top-level key of sol_id and bottom-level key of interaction_id.

    Returns:
        combined_demo (dict): A dictionary with (sol_id, interaction_id) as key, combined interaction string 
                              as value.
    """
    combined_demo = {}
    for sol_id, interactions in demo.items():
        for interaction_id, properties in interactions.items():
            pkg_chat = combine_interaction(properties['interaction'])
            combined_demo[(sol_id, interaction_id)] = pkg_chat

    return combined_demo


def generate_training_data(demo, filename):
    """
    Transforms the input dictionary into a list of dictionaries and writes it to a JSON file. Each dictionary in 
    the list represents an interaction and has 'entry_id', 'chat', and 'label' keys.

    Args:
        demo (dict): A dictionary with sol_id as keys and their properties as values. Each property is a 
                     dictionary of interactions.
        filename (str): The name of the file where the generated training data will be written.

    """
    combined_demo = []

    for idx, (sol_id, properties) in enumerate(demo.items()):
        entry = {}  # Init the dictionary for the current entry
        entry['entry_id'] = idx
        entry['chat'] = combine_interaction(properties['interaction'])
        entry['label'] = properties['label']
        combined_demo.append(entry)

    with open(filename, 'w') as file:
        json.dump(combined_demo, file, indent=4, sort_keys=True)


"""
    Automatic ChatGPT prompting class for reward dataset generation.
"""


# Prompter class
class GPTAutoPrompter:
    def __init__(self, prompt_rule: str, temperature: float = 0.7,
                 max_tokens: int = 100, top_p: float = 0.9,
                 presence_penalty: float = 0.1, frequency_penalty: float = 0.1,
                 label_mode: str = 'binary',
                 label_range: list = [-1, 1]):
        if label_mode not in ['binary', 'range']:
            raise ValueError("Argument 'label_mode' should be 'binary' or \
                             'range'.")

        self.prompt_rule = prompt_rule
        self.model_args = {'temperature': temperature,
                           'max_tokens': max_tokens,
                           'top_p': top_p,
                           'presence_penalty': presence_penalty,
                           'frequency_penalty': frequency_penalty}
        self.failed_pair = []
        self.init_label_mode(label_mode, label_range)

    def init_label_mode(self, label_mode, label_range):
        self.label_mode = label_mode
        if label_mode == 'range':
            self.output_rule = "You're asked to provide a decimal or integer score within the \
                           range of ({}, {}) based on the rules provided (can be equal to the \
                           boundary value). For a given interaction, the left boundary value \
                           refers to the most negative sample and the right boundary value \
                           refers to the most positive sample used in a dataset for training \
                           a reward model. Please don't be affected by the questions \
                           inside the interactions, for example the confidence score. Only consider if the whole interaction is a \
                           positive or negative sample for reward model training.".format(label_range[0],
                                                                                          label_range[1])
            self.label_range = label_range
        else:
            if label_range[0] >= label_range[1]:
                raise ValueError("Left boundary value should be smaller than right boundary value.")
            self.output_rule = "You're asked to provide an one-word output of 'positive' or \
                           'negative' based on the rules provided. This positive or negative \
                           indicates if the interaction provided should be considered as a \
                           positive sample or a negative sample in the dataset used for \
                           training a reward model. Please don't be affected by the questions \
                           inside the provided interactions, for example the confidence score. Only consider \
                           if the whole interaction is a positive or negative sample for reward model \
                           training."
            self.label_range = None

    def modify_prompt_instruction(self, new_prompt_rule: str):
        self.prompt_rule = new_prompt_rule + " " + self.output_rule

    @staticmethod
    def is_float_value(string):
        if string == "" or string == ".":
            return False
        pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)?\.?$'
        return re.match(pattern, string) is not None

    def fix_response(self, response: str):
        if self.label_mode == 'binary':
            response = response.lower()
            pos_str = 'positive';
            neg_str = 'negative'
            if pos_str in response and neg_str in response:
                return None, False
            elif pos_str in response:
                return pos_str, True
            elif neg_str in response:
                return neg_str, True
            else:
                return None, False
        else:
            left_value = self.label_range[0];
            right_value = self.label_range[1]
            if self.is_float_value(response):
                if response.endswith("."):
                    score = float(response[:-1])
                    if left_value <= score <= right_value:
                        return score, True
                    else:
                        return None, False
                else:
                    score = float(response)
                    if left_value <= score <= right_value:
                        return score, True
                    else:
                        return None, False
            else:
                return None, False

    def generate_response(self, sol_id, interaction_id, combined_chat: str, chat):
        label = None;
        proper_flag = False;
        counter = 0
        while not proper_flag:
            # If no expected output for 3 times of generation, break out and record
            if counter == 3:
                self.failed_pair.append((sol_id, interaction_id))
                break
            message = chat.ask(content=combined_chat,
                               model_args=self.model_args)
            response = message.to_dict()['content']
            label, proper_flag = self.fix_response(response)
            counter += 1
        return label

    def prompt(self, combined_demo: dict):
        prompt_labeled_result = []
        for idx, (key, combined_chat) in tqdm(enumerate(combined_demo.items()),
                                              total=len(combined_demo), desc="Processing"):
            (sol_id, interaction_id) = key
            # Warmup with rules and instructions, can modify if we have additional rules
            chat = Chat.create(name=f"sol_{sol_id}_interaction_{interaction_id}")
            _ = chat.ask(content=self.prompt_rule,
                         instruction=self.output_rule,
                         model_args=self.model_args)

            # Generate labels
            label = self.generate_response(sol_id=sol_id,
                                           interaction_id=interaction_id,
                                           combined_chat=combined_chat,
                                           chat=chat)
            if label is not None:
                labeled_dict = {"entry_id": idx,
                                "label": label,
                                "chat": combined_chat}
                prompt_labeled_result.append(labeled_dict)
        return prompt_labeled_result, self.failed_pair


"""
    Utility functions for interaction selection
"""


def pair_sol_with_interactions(
        interactions: Dict[int, Dict],
        solutions: Dict[int, Dict],
        qtype: str = 'ALL'
) -> Dict[int, List[tuple]]:
    """
    This function pairs interaction with its corresponding solution for each solution ID with a specific question type.

    Args:
        interactions: A dictionary where the keys are solution IDs and the values are
                      dictionaries of interactions associated with each solution ID.
        solutions: A dictionary where the keys are solution IDs and the values are
                   solutions associated with each solution ID.
        qtype: The question type to filter solutions by. If 'all', includes all question types.

    Returns:
        pairs_dict: A dictionary where keys are solution IDs and values are lists of tuples.
                    Each tuple contains a solution and its corresponding interaction, only for solutions of the given qtype.
    """

    # Find the common solution IDs between interactions and solutions_positive
    common_sol_ids = set(interactions.keys()) & set(solutions.keys())

    pairs_dict = {}

    # For each common solution ID
    for sol_id in common_sol_ids:

        # Check if the solution's qtype matches the given qtype, or if qtype is 'all'
        if qtype == "ALL" or solutions[sol_id]['qtype'] == qtype:

            # Init an empty list to store the solution-interaction pairs for this solution ID
            pairs_for_this_id = []

            # For each interaction associated with this solution ID
            for sub_dict in interactions[sol_id].items():
                # Create a pair of the solution and the interaction
                pair = (solutions[sol_id], sub_dict[1])

                # Add the pair to the list of pairs for this solution ID
                pairs_for_this_id.append(pair)

            # Add the list of pairs for this solution ID to the dictionary of all pairs
            pairs_dict[sol_id] = pairs_for_this_id

    return pairs_dict


def filter_by_condition_any(scores_dict, condition):
    """
    Returns a new dictionary containing only the entries where any score meets the given condition.

    Parameters:
    scores_dict: A dictionary containing the scores to be filtered.
    condition: A callable condition that will be checked against each score.

    Returns:
    A dictionary containing only the entries where any score meets the given condition.
    """
    return {sol_idx: scores for sol_idx, scores in scores_dict.items() if
            any(condition(score) for score in scores['scores'])}


def filter_by_condition_all(scores_dict, condition):
    """
    Returns a new dictionary containing only the entries where all scores meet the given condition.

    Parameters:
    scores_dict: A dictionary containing the scores to be filtered.
    condition: A callable condition that will be checked against each score.

    Returns:
    A dictionary containing only the entries where all scores meet the given condition.
    """
    return {sol_idx: scores for sol_idx, scores in scores_dict.items() if
            all(condition(score) for score in scores['scores'])}


def condition_length(scores, length):
    """
    Returns True if the length of the scores list is equal to the given length.

    Parameters:
    scores: A list of scores.
    length: The desired length.

    Returns:
    True if the length of the scores list is equal to the given length, False otherwise.
    """
    return len(scores) == length


def filter_by_conditions(scores_dict, score_conditions, filter_func, list_conditions=None):
    """
    Filters the scores dictionary by given conditions.

    Parameters:
    scores_dict: A dictionary containing the scores to be filtered.
    score_conditions: A list of callable conditions that will be checked against each score.
    filter_func: The filtering function to be used.
    list_conditions: A list of callable conditions that will be checked against the list of scores.

    Returns:
    A tuple of two dictionaries - the first one containing entries that meet all conditions, and the second one containing entries that do not meet all conditions.
    """
    if list_conditions is None:
        list_conditions = []

    passed_dict = {}  # This will contain entries that meet all conditions
    rest_dict = {}  # This will contain entries that do not meet all conditions

    for sol_idx, scores in scores_dict.items():
        score_conditions_met = all(filter_func({sol_idx: scores}, condition) for condition in score_conditions)
        list_conditions_met = all(condition(scores['scores']) for condition in list_conditions)

        if score_conditions_met and list_conditions_met:
            passed_dict[sol_idx] = scores
        else:
            rest_dict[sol_idx] = scores

    return passed_dict, rest_dict


def filter_by_score_diff(scores_dict, diff):
    """
    Filters the scores dictionary by the difference between the max and min scores.

    Parameters:
    scores_dict: A dictionary containing the scores to be filtered.
    diff: The difference threshold.

    Returns:
    A tuple of two dictionaries - the first one containing entries where the difference between the max and min scores is greater or equal to 'diff', and the second one where the difference is less than 'diff'.
    """
    passed_dict = {}
    rest_dict = {}

    for sol_idx, scores in scores_dict.items():
        if abs(max(scores['scores']) - min(scores['scores'])) >= diff:
            passed_dict[sol_idx] = scores
        else:
            rest_dict[sol_idx] = scores

    return passed_dict, rest_dict


def format_training_based_on_score(idx_score_pairs, sol_interaction_pair, score_diff=0.03):
    """
    This function formats the training data based on provided parameters.

    Parameters:
    idx_score_pairs: A dictionary containing pairs of indices and scores.
    sol_interaction_pair: A dictionary containing pairs of solutions and interactions.
    score_diff: A threshold for the difference between scores.

    Returns:
    formatted: A list of dictionaries with 'chosen' and 'rejected' interactions based on the scores.
    """
    formatted = []

    for key, score in idx_score_pairs.items():
        # Get the ground truth
        ref_interaction = combine_interaction(sol_interaction_pair[key][0][0]['interaction'])

        # Compare ground truth with every other interaction
        for idx in range(len(sol_interaction_pair[key])):
            formatted.append({
                'chosen': ref_interaction,
                'rejected': combine_interaction(sol_interaction_pair[key][idx][1]['interaction']),
            })

        for accept_idx, reject_idx in combinations(np.argsort(score['scores'])[::-1], 2):

            if score['scores'][accept_idx] - score['scores'][reject_idx] <= score_diff:
                continue

            formatted.append({
                'chosen': combine_interaction(sol_interaction_pair[key][accept_idx][1]['interaction']),
                'rejected': combine_interaction(sol_interaction_pair[key][reject_idx][1]['interaction'])
            })

    return formatted


def plot_frugal_dist(idx_score_dict):
    """
    Plots a histogram of the frugal scores distribution.

    Parameters:
    idx_score_dict: A dictionary containing pairs of indices and scores.

    Returns:
    None. This function displays a histogram plot.
    """
    frugal_scores = [score for scores in idx_score_dict.values() for score in scores['scores']]

    # Create a new figure
    fig = plt.figure()

    # Set the figure facecolor to white
    fig.patch.set_facecolor('white')

    # Add a new set of axes to the figure
    ax = fig.add_subplot(111)

    # Set the axes facecolor to white
    ax.set_facecolor('white')

    # Plot a histogram of the frugal scores
    ax.hist(frugal_scores, bins=20, alpha=0.5, color='blue')
    ax.set_xlabel('Frugal Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Frugal Scores for all types of questions')

    # Set x-axis ticks to be every 0.01
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 0.05))

    ax.grid(True)

    # Display the figure
    plt.show()


"""
    Utility functions for NLI Reward Dataset Generation
"""


def populate_dict(pairs):
    """
    Populates a dictionary with sol_id as keys and a list of interaction_ids as values.
    """
    result_dict = {}
    for sol_id, interaction_id in pairs:
        if sol_id in result_dict:
            result_dict[sol_id].append(interaction_id)
        else:
            result_dict[sol_id] = [interaction_id]
    return result_dict


def create_chosen_reject_pair(chosen_dict, reject_dict, interactions, solutions):
    """
    Pair reject interactions with chosen interactions,
    as well as pairs where 'chosen' is solution and 'rejected' is interactions in reject_dict.
    """
    rm_training_pairs = []

    for sol_id, reject_interaction_ids in reject_dict.items():
        if sol_id in solutions:
            for reject_interaction_id in reject_interaction_ids:
                rm_training_pairs.append({
                    "chosen": combine_interaction(solutions[sol_id]['interaction']),
                    "rejected": combine_interaction(interactions[sol_id][reject_interaction_id]['interaction']),
                })

        if sol_id in chosen_dict:
            chosen_interaction_ids = chosen_dict[sol_id]
            for reject_interaction_id in reject_interaction_ids:
                for chosen_interaction_id in chosen_interaction_ids:
                    rm_training_pairs.append({
                        "chosen": combine_interaction(interactions[sol_id][chosen_interaction_id]['interaction']),
                        "rejected": combine_interaction(interactions[sol_id][reject_interaction_id]['interaction']),
                    })

    return rm_training_pairs


def transform_interactions(samples):
    """Index samples by sol_id and interaction_id, and also count the roles separately.

    Args:
        samples (dict): A list of samples, each containing a sol_id and an interaction_id.

    Returns:
        dict, dict: Two nested dictionaries where the first level of keys are sol_id values and the second level of keys are interaction_id values.
                    The first dictionary contains the samples and the second dictionary contains the role counts.
    """

    # Initialize the dictionaries
    indexed_samples = {}
    role_freq = {}

    for sol_id, pairs in samples.items():
        for interaction_id, sample in pairs.items():

            # If this sol_id is not already in the dictionary, add it
            if sol_id not in indexed_samples:
                indexed_samples[sol_id] = {}
                role_freq[sol_id] = {}

            # Add the sample and role count to the dictionaries, indexed by sol_id and interaction_id
            indexed_samples[sol_id][interaction_id] = sample
            role_freq[sol_id][interaction_id] = count_role_freq(sample)

    # Convert role_freq to a DataFrame to facilitate analysis
    role_freq_df = pd.DataFrame.from_dict({(i, j): role_freq[i][j]
                                           for i in role_freq.keys()
                                           for j in role_freq[i].keys()},
                                          orient='index')

    return indexed_samples, role_freq_df