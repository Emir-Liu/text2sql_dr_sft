import os

### path config
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data pre-process configuration
DATA_PATH = os.path.join(ROOT_PATH, "data")

SQL_DATA_INFO = [{
    "data_source": "CSpider",
    "train_file": ["train.json"],
    "dev_file": ["dev.json"],
    "train_tables_file": "tables.json",
    "dev_tables_file": "tables.json",
    "db_id_name": "db_id",
    "output_name": "query",
    "is_multiple_turn": False,
},{
    "data_source": "spider",
    "train_file": ["train_spider.json","train_others.json"],
    "dev_file": ["dev.json"],
    "train_tables_file": "tables.json",
    "dev_tables_file": "tables.json",
    "db_id_name": "db_id",
    "output_name": "query",
    "is_multiple_turn": False,
}
]

INSTRUCTION_PROMPT = """\
I want you to act as a SQL terminal in front of an example database, \
you need only to return the sql command to me.Below is an instruction that describes a task, \
Write a response that appropriately completes the request.\n"
Instruction:\n{}\n"""

INPUT_PROMPT = "###Input:\n{}\n\n###Response:"

TOTAL_PROMPT = """<s>[INST]<<SYS>>{}<</SYS>>{}[/INST]{} </s>"""

PART_PROMPT = """<s>[INST]<<SYS>>{}<</SYS>>{}[/INST]"""

# training configuration
ADAPTER_PATH = os.path.join(ROOT_PATH, "model/adapter")

PER_DEVICE_TRAIN_BATCH_SIZE = 4
EVAL_STEPS = 1000
SAVE_STEPS = 5
LOGGING_STEPS = 1
EVALUATION_STRATEGY='steps'
MAX_STEPS= 10
NUM_TRAIN_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 16
GRADIENT_CKPT=True
MAX_GRAD_NORM = 0.3
LR_SCHEDULER_TYPE = 'cosine_with_restarts'
WARMUP_STEPS = 1000
LEARNING_RATE = 2E-4
TRAIN_MODEL_PATH = '.'

# MODELS_PARENT_PATH = "/home/ymLiu/model"
# DEFAULT_FT_MODEL_NAME = "CodeLlama-7b-Instruct"
# MODEL_PATH = os.path.join(MODELS_PARENT_PATH, DEFAULT_FT_MODEL_NAME)
# model configuration
MODEL_PATH = "/home/ymLiu/model/CodeLlama-7b-Instruct"

# predict sql cmd path
PRED_PATH = os.path.join(ROOT_PATH, "pred")

# merged model path
MERGED_MODEL_PATH = os.path.join(ROOT_PATH, "model/merged_model")


# training model path

# MERGED_MODELS = os.path.join(ROOT_PATH, "output/merged_models")


"""
PREDICTED_DATA_PATH = os.path.join(
    ROOT_PATH, "data/eval_data/dev_sql.json"
)
PREDICTED_OUT_FILENAME = "pred_sql.sql"

OUT_DIR = os.path.join(ROOT_PATH, "output/")

## model constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


LOG_FILE_NAME = "trainer_log.jsonl"

# head_state_dict,model save name
VALUE_HEAD_FILE_NAME = "value_head.bin"

# output ,finetuning_args save_to_json name
FINETUNING_ARGS_NAME = "finetuning_args.json"

#  when prepare_model_for_training ,layer_norm_names
LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp"]
EXT2TYPE = {"csv": "csv", "json": "json", "jsonl": "json", "txt": "text"}
"""

# text2sql dataset information for processing sql data
"""
SQL_DATA_INFO = [{
    "data_source": "spider",
    "train_file": ["train_spider.json", "train_others.json"],
    "dev_file": ["dev.json"],
    "train_tables_file": "tables.json",
    "dev_tables_file": "tables.json",
    "db_id_name": "db_id",
    "output_name": "query",
    "is_multiple_turn": False,
}]
"""




# INSTRUCTION_ONE_SHOT_PROMPT = """\
# I want you to act as a SQL terminal in front of an example database. \
# You need only to return the sql command to me. \
# First, I will show you few examples of an instruction followed by the correct SQL response. \
# Then, I will give you a new instruction, and you should write the SQL response that appropriately completes the request.\
# \n### Example1 Instruction:
# The database contains tables such as employee, salary, and position. \
# Table employee has columns such as employee_id, name, age, and position_id. employee_id is the primary key. \
# Table salary has columns such as employee_id, amount, and date. employee_id is the primary key. \
# Table position has columns such as position_id, title, and department. position_id is the primary key. \
# The employee_id of salary is the foreign key of employee_id of employee. \
# The position_id of employee is the foreign key of position_id of position.\
# \n### Example1 Input:\nList the names and ages of employees in the 'Engineering' department.\n\
# \n### Example1 Response:\nSELECT employee.name, employee.age FROM employee JOIN position ON employee.position_id = position.position_id WHERE position.department = 'Engineering';\
# \n###New Instruction:\n{}\n"""
