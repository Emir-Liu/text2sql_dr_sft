"""merge lora and base model together"""
# from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import json

from typing import List
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)


from peft import PeftModel

from text2sql.model_operator import ModelOperator



if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     use_fast=model_args.use_fast_tokenizer,
    #     split_special_tokens=model_args.split_special_tokens,
    #     padding_side="right",  # training with left-padded tensors in fp16 precision may cause overflow
    #     **config_kwargs
    # )
    model_operator = ModelOperator()

    base_model = ModelOperator().model
    base_tokenizer = ModelOperator().tokenizer

    lora_ckpt = './model/adapter/checkpoint-2000'
    lora_model = PeftModel.from_pretrained(base_model, lora_ckpt)

    merge_model = lora_model.merge_and_unload()

    merge_model_path = './model/merged_model/CS_2000'
    merge_model.save_pretrained(merge_model_path, max_shard_size='5GB')

    base_tokenizer.save_pretrained(merge_model_path)




