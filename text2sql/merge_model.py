"""merge lora and base model together"""
# from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import json

import argparse

from typing import List
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)


from peft import PeftModel

from text2sql.model_operator import ModelOperator
from configs.config import MERGED_MODEL_PATH, ADAPTER_PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_name", help="adapter name", default=None
    )
    parser.add_argument(
        '--base_model_name_or_path', help="base model name or path", default=None
    )
    parser.add_argument(
        '--merged_model_name', help="merged model name", default=None
    )
    args = parser.parse_args()
    adapter_name = args.adapter_name
    base_model = args.base_model_name_or_path
    merged_model_name = args.merged_model_name
    
    print(f'adapter_name:{adapter_name}\nbase_model:{base_model}\nmerged_model_name:{merged_model_name}')
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     use_fast=model_args.use_fast_tokenizer,
    #     split_special_tokens=model_args.split_special_tokens,
    #     padding_side="right",  # training with left-padded tensors in fp16 precision may cause overflow
    #     **config_kwargs
    # )

    model_operator = ModelOperator(model_name_or_path=base_model)
    model_operator.load_model_and_tokenizer()
    base_model = model_operator.model
    base_tokenizer = model_operator.tokenizer

    lora_ckpt = os.path.join(ADAPTER_PATH, adapter_name)
    print(f'lora_ckpt:{lora_ckpt}')
    # lora_ckpt = './model/adapter/checkpoint-2000'
    lora_model = PeftModel.from_pretrained(base_model, lora_ckpt)

    merge_model = lora_model.merge_and_unload()

    merge_model_path = './model/merged_model/CS_2000'
    merged_model_full_path = os.path.join(MERGED_MODEL_PATH, merged_model_name)
    merge_model.save_pretrained(merged_model_full_path, max_shard_size='5GB')

    base_tokenizer.save_pretrained(merge_model_path)




