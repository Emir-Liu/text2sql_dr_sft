# import os
# import sys
# import argparse
# import time

# ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f'ROOT_PATH:{ROOT_PATH}')
# sys.path.append(ROOT_PATH)

# from trl import SFTTrainer

# from text2sql.model_operator import ModelOperator, DatasetOperator
# # from text2sql.training_config import peft_config, training_arguments
# from text2sql.training_config import get_lora_and_train_config
# from configs.config import DATA_PATH, ADAPTER_PATH


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dataset", help="dataset name", default=False
#     )
#     parser.add_argument(
#         '--base_model_name_or_path', help="base model name or path", default=None
#     )
#     parser.add_argument(
#         '--deepspeed'
#     )

#     parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
#     args = parser.parse_args()
#     dataset_name = args.dataset
#     base_model = args.base_model_name_or_path

#     base_model_part_name = base_model.split('/')[-1]


#     # load model
#     model_operator = ModelOperator(model_name_or_path=base_model)

#     model_operator.load_model_and_tokenizer()

#     model = model_operator.model
#     tokenizer = model_operator.tokenizer

#     # load dataset
#     train_file = os.path.join(DATA_PATH, dataset_name, "text2sql_train.json")
#     dev_file = os.path.join(DATA_PATH, dataset_name, "text2sql_dev.json")

#     train_dataset_loader = DatasetOperator().load_and_get_dataset_loader(train_file)
#     dev_dataset_loader = DatasetOperator().load_and_get_dataset_loader(dev_file)
#     adapter_name=f'{base_model_part_name}_{dataset_name}_{time.strftime("%y%m%d_%H%M%S", time.localtime())}'
#     peft_config, training_arguments = get_lora_and_train_config(adapter_name)
#     # a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. 
#     # This might lead to some unexpected behaviour due to overflow issues when training a model in half-precision. 
#     # You might consider adding `tokenizer.padding_side = 'right'` to your code.
#     tokenizer.padding_side = 'right'     
#     trainer = SFTTrainer(
#         model = model, 
#         train_dataset = train_dataset_loader, 
#         eval_dataset = dev_dataset_loader, 
#         dataset_text_field = 'text', 
#         peft_config = peft_config, 
#         tokenizer = tokenizer,
#         args = training_arguments,
#     )

#     # start training
#     trainer.train()

#     # save model
#     trainer.model.save_pretrained(os.path.join(ADAPTER_PATH, adapter_name),)

# ref link: https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

from trl.commands.cli_utils import SFTScriptArguments, TrlParser


from datasets import load_dataset

from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    print(f'args:{args}\ntraining_args:{training_args}\nmodel_config:{model_config}\n')

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)