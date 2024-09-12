import os
import sys
import argparse
import time

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)

from trl import SFTTrainer

from text2sql.model_operator import ModelOperator, DatasetOperator
# from text2sql.training_config import peft_config, training_arguments
from text2sql.training_config import get_lora_and_train_config
from configs.config import DATA_PATH, ADAPTER_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="dataset name", default=False
    )
    parser.add_argument(
        '--base_model_name_or_path', help="base model name or path", default=None
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    base_model = args.base_model_name_or_path

    base_model_part_name = base_model.split('/')[-1]


    # load model
    model_operator = ModelOperator(model_name_or_path=base_model)

    model_operator.load_model_and_tokenizer()

    model = model_operator.model
    tokenizer = model_operator.tokenizer

    # load dataset
    train_file = os.path.join(DATA_PATH, dataset_name, "text2sql_train.json")
    dev_file = os.path.join(DATA_PATH, dataset_name, "text2sql_dev.json")

    train_dataset_loader = DatasetOperator().load_and_get_dataset_loader(train_file)
    dev_dataset_loader = DatasetOperator().load_and_get_dataset_loader(dev_file)
    adapter_name=f'{base_model_part_name}_{dataset_name}_{time.strftime("%y%m%d_%H%M%S", time.localtime())}'
    peft_config, training_arguments = get_lora_and_train_config(adapter_name)
    # a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. 
    # This might lead to some unexpected behaviour due to overflow issues when training a model in half-precision. 
    # You might consider adding `tokenizer.padding_side = 'right'` to your code.
    tokenizer.padding_side = 'right'     
    trainer = SFTTrainer(
        model = model, 
        train_dataset = train_dataset_loader, 
        eval_dataset = dev_dataset_loader, 
        dataset_text_field = 'text', 
        peft_config = peft_config, 
        tokenizer = tokenizer,
        args = training_arguments,
    )

    # start training
    trainer.train()

    # save model
    trainer.model.save_pretrained(os.path.join(ADAPTER_PATH, adapter_name),)