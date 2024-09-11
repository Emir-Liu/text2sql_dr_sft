import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'ROOT_PATH:{ROOT_PATH}')
sys.path.append(ROOT_PATH)


from text2sql.model_operator import ModelOperator, DatasetOperator
from text2sql.training_config import peft_config, training_arguments
from configs.config import DATA_PATH, TRAIN_MODEL_PATH



from trl import SFTTrainer

if __name__ == '__main__':
    # load model
    model_operator = ModelOperator()

    model_operator.load_model_and_tokenizer()

    model = model_operator.model
    tokenizer = model_operator.tokenizer

    # load dataset
    train_file = os.path.join(DATA_PATH, "text2sql_train.json")
    dev_file = os.path.join(DATA_PATH, "text2sql_dev.json")

    train_dataset_loader = DatasetOperator().load_and_get_dataset_loader(train_file)
    dev_dataset_loader = DatasetOperator().load_and_get_dataset_loader(dev_file)


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
    trainer.model.save_pretrained(TRAIN_MODEL_PATH)