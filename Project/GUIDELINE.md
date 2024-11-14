# Text Generation Guideline
We provide an easy-to-follow environment installation and script execution guideline, this guildline supports Linux/Mac OS. 

For Windows users, please follow the anaconda official installation guideline and skip the first 2 steps. Then please execute the shell scripts in Git Bash.

[*Back to PROJECT_INTRO*](./PROJECT_INTRO.md)

## Install conda environment
(Or use your own newly created/skip this step if only for testing)

1. *Install Miniconda*
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"
```

2. *Create new conda environment named "nlp" (or other names)*
```
conda create -n nlp python=3.10
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate nlp
```

3. *Install all required packages under `requirements.txt`*
```
pip install -r requirements.txt
```

## Reproduce the Results
### Download Models
Fine-tuned adapter-merged LLaMA-7B base model: https://drive.google.com/drive/folders/1zfm2jrW2pHriiWhivWrebVELeiNKyf0-?usp=sharing

Fine-tuned adapter-merged reward base model: https://drive.google.com/drive/folders/1yH3BIDqnOIgZcj57N6q1pCrZdQzlgsRh?usp=sharing

For text generation, please download the all LLaMA-7B files in folder and place them under a folder, and modify the path of `base_model` argument correspondingly in `generate.sh`. `models/ppo-fine-tuned-ethical-frugal` has the adapter configs and weights that fine-tuned on our training dataset, it should be used with the downloaded base model in advance. Default path to `--base_model` is `models/lora-alpaca-adapter_merged`.

### Fast-Running Prepared Evaluation
If you just want to check the evaluation metric, you can use the `prompt_processed.json`, `answers_ikun.json` (prediction) under `./dataset`, and `prompts.json` that are already processed. Then please run:
```
bash eval.sh
```

You will see the evaluation metrics in the command line, plus a saved JSON file under `./dataset` with a filename of the prediction file's filename + *_evaluate* showing the metrics for each generated sample.

### A Complete Reproduction
The above steps are for completely reproducing the results. Please follow the processing and generation configurations in the shell scripts.

**Prompt Preprocessing**:

Since the testing prompts file `prompts.json` is not naturally provided in the desired prompting form, we'll first need to convert it into a standard prompting file for generation and evaluation:
```
bash convert.sh
```

Please fix the source and target file directory as you need, if you're running the generation for the first time, please always set `--initial_conversion` in `convert.sh`.

**Text Generation**:

For a quick text generation, you can try on a Nvidia graphic card supported device (we recommend GPU RAM >= 16GB) by execute the following (you can modify generation settings also in the shell script below):
```
bash generate.sh
```

Default text generation is carried out on the converted prompts you generated in the *prompt preprocessing* step. By default, you will get a JSON file named `prompt_result` under `./dataset`, in the form as required in the final report specification. 

**Metric Evaluation**:

Finally, please check and fix all path and file arguments in `eval.sh`, to be consistent with your generations in the above two steps. And similarly, run:
```
bash eval.sh
```

And you can get the evaluation metric results as stated in the *Fast-Running Prepared Evaluation*.

### Further Fine-tuning:

The training and fine-tuning process is a complicated and recursive process, if you want to know our detailed training pipeline, please contact us by email.
