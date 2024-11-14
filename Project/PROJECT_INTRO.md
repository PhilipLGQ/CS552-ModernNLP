# CS552 Final Project: Educational Chatbot Model (Text Generation)
## Project Introduction
This repository stores our course project of building up an educational chatbot (text generation) model. Our chatbot model is based the [LLaMA-7B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) fine-tuned with [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) strategy in [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) with a fine-tuned [DeBERTaV3-base](https://huggingface.co/microsoft/deberta-v3-base) model as the reward model. 

## File Structure
* `dataset/`: folder saving all datasets we used for model fine-tuning, text generation and mode evaluation.
* `src/`: folder saving all source code of training/testing generative model, reward model, and RLHF PPO.
* `models/`: folder saving the fine-tuned models and model configurations.
* `templates/`: folder saving the prompting template for the generative model. You can adapt your own prompting template following the form of [`alpaca.json`](./templates/alpaca.json)
* `notebook/`: folder saving mainly the data processing steps and helper functions used.
* `gen_script_ikun.json`: an JSON format copy of [`generate.py`](./src/generate.py), used to generate the prompting results.
* `GUIDELINE.md`: an easy-to-follow guideline to test our chatbot model.
* `README.md`: the original project requirement description provided.
* `TEAM_CONTRIBUTION.md`: a brief summary of project contributions carried out by each team member
* `prompts.json`: the original "gold label" prompt testing dataset
* `eval.sh`: shell script to run metric evaluation
* `convert.sh`: shell script to process `prompts.json` into our standard prompting form
* `generate.sh`: shell script to run text generation on fine-tuned LLaMA-7B model
* `train.sh`: shell script to run supervised fine-tuning on LLaMA-7B model

*Milestone 2 Continuous in Milestone 3*
* `evaluate.py`: for reward model testing as in Milestone 2
* `model.py`: (same as above)
* `m2_reward_dataset_example.json`: (same as above) 

## Prerequisites
Please refer to [`requirements.txt`](./requirements.txt) to run fully the project.

## Quick Start
Please follow the instructions as demonstrated in [`GUIDELINE.md`](./GUIDELINE.md).

Furthur, if you want to test the reward model as in Milestone 2, please refer to `evaluate.py` and `model.py`.

## Authors:
* Guanqun Liu (guanqun.liu@epfl.ch)
* Yixuan Xu (yixuan.xu@epfl.ch)
* Hao Zhao (hao.zhao@epfl.ch)

