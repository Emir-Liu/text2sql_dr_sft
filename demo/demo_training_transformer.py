"""Lora training with transformer package
"""
import os
import sys
import time

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)

# import deepspeed
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
import numpy as np

from text2sql.model_operator import ModelOperator, DatasetOperator
from configs.config import DATA_PATH, ADAPTER_PATH

# from text2sql.seq2seq_trainer import Seq2SeqTrainer

if __name__ == '__main__':

    dataset_name = 'sample_data'
    base_model = '/home/ymLiu/model/CodeLlama-7b-Instruct'
    base_model_part_name = base_model.split('/')[-1]

    # load model
    model_operator = ModelOperator(model_name_or_path=base_model)

    model_operator.load_model_and_tokenizer()

    model = model_operator.model
    tokenizer = model_operator.tokenizer
    tokenizer.padding_side = 'right'
    # load dataset
    train_file = os.path.join(DATA_PATH, dataset_name, "text2sql_train.json")
    dev_file = os.path.join(DATA_PATH, dataset_name, "text2sql_dev.json")

    train_dataset_loader = DatasetOperator().load_and_get_dataset_loader(train_file)
    dev_dataset_loader = DatasetOperator().load_and_get_dataset_loader(dev_file)
    max_input_length = 128
    max_target_length = 128
    def preprocess_function(examples):
        inputs = examples['question']
        targets = examples['ans']
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(f'train_dataset_loader:{train_dataset_loader}')

    # tokenized_train_datasets = train_dataset_loader.map(lambda x:
    #     {
    #         'input_ids': tokenizer(x['question'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['input_ids'],
    #         'attention_mask': tokenizer(x['question'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['attention_mask'],
    #         'labels': tokenizer(x['ans'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['input_ids']
    #     }
    # )
    tokenized_train_datasets = train_dataset_loader.map(preprocess_function, batched=True)
    # for item in tokenized_train_datasets:
    #     print(f'item tokenized_train_datasets:{item}')
    # print(f'tokenized_train_datasets:{tokenized_train_datasets}')
    
    # tokenized_dev_datasets = dev_dataset_loader.map(lambda x:
    #     {
    #         'input_ids': tokenizer(x['question'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['input_ids'],
    #         'attention_mask': tokenizer(x['question'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['attention_mask'],
    #         'labels': tokenizer(x['ans'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['input_ids']
    #     }
    # )
    tokenized_dev_datasets = dev_dataset_loader.map(preprocess_function, batched=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir = '.',
        num_train_epochs=1,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        use_cpu=False
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    adapter_name=f'{base_model_part_name}_{dataset_name}_{time.strftime("%y%m%d_%H%M%S", time.localtime())}'
    trainer = Seq2SeqTrainer(
        model = model,
        train_dataset = tokenized_train_datasets,
        eval_dataset = tokenized_dev_datasets,
        data_collator = data_collator,
        args=training_args,
    )
    print('开始训练')
    trainer.train()

    # # pred
    # import torch
    # input_ids = tokenized_train_datasets['input_ids'][0]
    # labels = tokenized_train_datasets['labels'][0]
    # print(f'input ids:{input_ids}')
    # print(f'labels:{labels}')

    # response = model.generate(torch.tensor(input_ids).to('cuda:0'), max_length=300)
    # print(f'response:{response}')

    # print(f'len input:{len(input_ids)}')
    # print(f'labels:{len(labels)}')
    # print(f'response:{len(response[0])}')

    