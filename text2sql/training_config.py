"""set training and lora configuration
"""
import os
import time

from peft import LoraConfig
from transformers import TrainingArguments



from configs.config import (
    ADAPTER_PATH,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    EVAL_STEPS,
    SAVE_STEPS,
    LOGGING_STEPS,
    EVALUATION_STRATEGY,
    # MAX_STEPS,
    NUM_TRAIN_EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CKPT,
    MAX_GRAD_NORM,
    LR_SCHEDULER_TYPE,
    # WARMUP_STEPS,
    LEARNING_RATE,
)

def get_lora_and_train_config(adapter_name:str=time.strftime("%y%m%d_%H%M%S", time.localtime()),deepspeed_configuration=None, local_rank=0):
    # lora configuration
    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        # target_modules=["gate_proj", "down_proj", "up_proj"],
        target_modules = ['q_proj','v_proj'], # attention the layer
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


    # training configuration
    training_arguments = TrainingArguments(
        output_dir = os.path.join(ADAPTER_PATH, adapter_name),
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        eval_steps=EVAL_STEPS,
        save_steps = SAVE_STEPS,
        logging_steps = LOGGING_STEPS,
        eval_strategy = EVALUATION_STRATEGY,
        # max_steps = MAX_STEPS,
        num_train_epochs = NUM_TRAIN_EPOCHS,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing = GRADIENT_CKPT,
        max_grad_norm =  MAX_GRAD_NORM,
        learning_rate = LEARNING_RATE,
        lr_scheduler_type = LR_SCHEDULER_TYPE,
        deepspeed = deepspeed_configuration,
        local_rank = local_rank,
        # generation_config = None,
        # warmup_steps = WARMUP_STEPS,
        
    )

    return peft_config, training_arguments


