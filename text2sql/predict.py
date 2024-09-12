"""load model and predict with prompt
"""

import os
import sys
import json
import argparse

import re

from tqdm import tqdm

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)

from configs.config import PART_PROMPT, INSTRUCTION_PROMPT, DATA_PATH, PRED_PATH
from text2sql.model_operator import ModelOperator, DatasetOperator

def predict_single(org_data:dict = None) -> str:
    if org_data is None:
        org_data = {
            "db_id": "concert_singer",
            "instruction": "I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\"\nInstruction:\nconcert_singer contains tables such as stadium, singer, concert, singer_in_concert. Table stadium has columns such as Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average. Stadium_ID is the primary key.\nTable singer has columns such as Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male. Singer_ID is the primary key.\nTable concert has columns such as concert_ID, concert_Name, Theme, Stadium_ID, Year. concert_ID is the primary key.\nTable singer_in_concert has columns such as concert_ID, Singer_ID. concert_ID is the primary key.\nThe Stadium_ID of concert is the foreign key of Stadium_ID of stadium.\nThe Singer_ID of singer_in_concert is the foreign key of Singer_ID of singer.\nThe concert_ID of singer_in_concert is the foreign key of concert_ID of concert.\n\n",
            "input": "###Input:\n我们有多少歌手？\n\n###Response:",
            "english_input": "###Input:\nHow many singers do we have?\n\n###Response:",
            "output": "SELECT count(*) FROM singer",
            "history": []
        }
    model_operator = ModelOperator()
    model_operator.load_model_and_tokenizer()
    # model = model_operator.model
    # tokenizer = model_operator.tokenizer

    instructs = org_data['instruction']
    inputs = org_data['english_input']

    total_prompt = PART_PROMPT.format(INSTRUCTION_PROMPT.format(instructs), inputs)

    response = model_operator.generate(content=total_prompt)
    print(f'response:{response}')

def inference(model_operator, dataset_loader):
    res_list = []
    # for item in dataset_loader:
    for item in tqdm(dataset_loader):
        instructs = item['instruction']
        inputs = item['input']

        total_prompt = PART_PROMPT.format(INSTRUCTION_PROMPT.format(instructs), inputs)
        response = model_operator.generate(total_prompt, temperature=0)
        res_list.append(response)
        # break
    return res_list

def predict_dataset(predict_file_name:str, base_model_name_or_path:str, adapter_name:str, dataset_name:str):
    # predict_path = './pred/model.sql'
    predict_path = os.path.join(PRED_PATH, predict_file_name)
    model_operator = ModelOperator(model_name_or_path=base_model_name_or_path)

    model_operator.load_model_and_tokenizer()

    # load dataset
    dev_file = os.path.join(DATA_PATH,dataset_name, "text2sql_dev.json")

    dev_dataset_loader = DatasetOperator().load_and_get_dataset_loader(dev_file)

    res = inference(model_operator, dev_dataset_loader)

    with open(predict_path, 'w') as f:
        for p in res:
            try:
                p = p.replace('\n', '')
                f.write(p + '\n')
            except:
                f.write('Invalid Output\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="dataset name", default=False
    )
    parser.add_argument(
        '--base_model_name_or_path', help="base model name or path", default=None
    )
    parser.add_argument(
        '--adapter_name', help="base model name or path", default=None
    )
    parser.add_argument(
        '--pred_file_name', help='predict result file name', default=None
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    base_model = args.base_model_name_or_path
    adapter_name = args.adapter_name
    pred_file_name = args.pred_file_name

    predict_dataset(predict_file_name=pred_file_name, base_model_name_or_path=base_model, adapter_name=adapter_name, dataset_name=dataset_name)