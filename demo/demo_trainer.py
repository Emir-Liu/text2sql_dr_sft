import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)

import torch
from text2sql.model_operator import ModelOperator, DatasetOperator
# from text2sql.training_config import get_lora_and_train_config
from configs.config import DATA_PATH, ADAPTER_PATH
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, Trainer

dataset_name= 'sample_data'

train_file = os.path.join(DATA_PATH, dataset_name, "text2sql_train.json")
dev_file = os.path.join(DATA_PATH, dataset_name, "text2sql_dev.json")

train_dataset_loader = DatasetOperator().load_and_get_dataset_loader(train_file)
dev_dataset_loader = DatasetOperator().load_and_get_dataset_loader(dev_file)

print(f'train_dataset_loader:{train_dataset_loader}')

max_input_length = 2
max_target_length = 2
# source_lang = "en"
# target_lang = "ro"

# model_checkpoint = '/home/ymLiu/model/CodeLlama-7b-Instruct'
model_checkpoint = '/home/it/model/chatglm3-6b'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
# tokenizer.pad_token_id = 0

def preprocess_function(examples):
    # inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    # targets = [ex[target_lang] for ex in examples["translation"]]
    inputs = [ ex for ex in examples['text']]
    targets = [ ex for ex in examples['ans']]

    model_inputs = tokenizer(examples['text'], max_length=max_input_length, truncation=True, padding=True, return_tensors='pt')
    # 'input_ids': tokenizer(x['question'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['input_ids'],
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer( examples['ans'], max_length=max_target_length, truncation=True, padding=True, return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


data = preprocess_function(train_dataset_loader[:2])
print(f'data:{data}')
print(f'len input ids:{len(data['input_ids'])}')
print(f'len labels:{len(data['labels'])}')



tokenized_train_datasets = train_dataset_loader.map(preprocess_function, batched=True)
# tokenized_train_datasets = train_dataset_loader.map(lambda x:
#     {
#         'input_ids': tokenizer(x['question'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['input_ids'],
#         'attention_mask': tokenizer(x['question'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['attention_mask'],
#         'labels': tokenizer(x['ans'], max_length=2048, truncation=True, padding=True, return_tensors='pt')['input_ids']
#     }
# )

# print('tokenized_datasets:')
# for tmp_tokenized_datasets in tokenized_datasets[0:2]:
#     print(f'{tokenized_datasets}')

model = AutoModelForCausalLM.from_pretrained(model_checkpoint,torch_dtype = torch.float16, trust_remote_code=True)

batch_size = 1
args = Seq2SeqTrainingArguments(
    "hello",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# for item in tokenized_train_datasets:
#     print(f'item tokenized_datasets:{item}')
# print(f'tokenized_train_datasets:{tokenized_train_datasets}')

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_train_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

