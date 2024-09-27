"""model operation include: loading model, save model etc.
"""

import os
import sys
import json

from typing import List

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"ROOT_PATH:{ROOT_PATH}")
sys.path.append(ROOT_PATH)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# unknown
from datasets import Dataset


from configs.config import MODEL_PATH, DATA_PATH, TOTAL_PROMPT, PART_PROMPT, ANS_PROMPT


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelOperator:

    def __init__(self, model_name_or_path=MODEL_PATH):
        self.model = None
        self.tokenizer = None
        self.model_name_or_path = model_name_or_path

    def load_model_and_tokenizer(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
        )
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = 0

    def encode(self, content: str):
        input_ids = self.tokenizer.encode(content, return_tensors="pt").to(device)
        # print(f'input_ids:{input_ids}')
        return input_ids

    def decode(self, ids) -> str:
        content = self.tokenizer.decode(ids, skip_special_tokens=True)
        return content

    def generate(self, content: str):
        input_ids = self.encode(content)

        prompt_len = len(input_ids[0])

        output_ids = self.model.generate(input_ids, pad_token_id=0, max_length=1024)

        part_output_ids = output_ids.tolist()[0][prompt_len:]
        output = self.decode(part_output_ids)
        return output


class DatasetOperator:

    def load_datasetfile(self, file_path) -> List[dict]:
        """load dataset file

        Args:
            file_path(str): dataset file path

        Returns:
            List[dict]
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # add text key
        new_data = []
        for tmp_data in data:
            tmp_data["text"] = TOTAL_PROMPT.format(
                tmp_data["instruction"], tmp_data["input"], tmp_data["output"]
            )
            new_data.append(tmp_data)

            tmp_data["question"] = PART_PROMPT.format(
                tmp_data["instruction"], tmp_data["input"]
            )
            tmp_data["ans"] = ANS_PROMPT.format(tmp_data["output"])

        return new_data

    def get_dataset_loader(self, data):
        transform_data = {key: [dic[key] for dic in data] for key in data[0]}
        dataset_loader = Dataset.from_dict(transform_data)
        return dataset_loader

    def load_and_get_dataset_loader(self, file_path):
        data = self.load_datasetfile(file_path)
        dataset_loader = self.get_dataset_loader(data)
        return dataset_loader


if __name__ == "__main__":
    # load model
    model_operator = ModelOperator()

    model_operator.load_model_and_tokenizer()

    model = model_operator.model

    # load dataset
    train_file = os.path.join(DATA_PATH, "text2sql_train.json")
    dev_file = os.path.join(DATA_PATH, "text2sql_dev.json")

    train_dataset_loader = DatasetOperator().load_and_get_dataset_loader(train_file)

    for tmp_item in train_dataset_loader:
        print(f"train dataset loader item:{tmp_item}")
