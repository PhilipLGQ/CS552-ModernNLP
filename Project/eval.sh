#!/bin/bash

python ./src/eval.py --root-path "/content/drive/MyDrive/project-m3-ikun-drive" \
                    --batch-size 16 \
                    --input-prompt-file "prompts.json" \
                    --model-prediction-file "dataset/answers_ikun.json" \
