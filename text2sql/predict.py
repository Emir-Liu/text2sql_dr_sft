"""load model and predict with prompt
"""

import os
import sys
import json
import argparse

from tqdm import tqdm

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)

from configs.config import PART_PROMPT, INSTRUCTION_PROMPT
from text2sql.model_operator import ModelOperator


if __name__ == '__main__':

    model_operator = ModelOperator()
    model_operator.load_model_and_tokenizer()
    model = model_operator.model
    tokenizer = model_operator.tokenizer

    instructs = 'hello'
    inputs = 'hello'

    total_prompt = PART_PROMPT.format(INSTRUCTION_PROMPT.format(instructs), inputs)

    response = model_operator.generate(content=total_prompt)
    print(f'response:{response}')


