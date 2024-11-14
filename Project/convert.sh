#!/bin/bash

python ./src/convert.py --previous_path './prompts.json' \
                  --save_path './dataset/prompts_processed.json' \
                  --initial_conversion
                  