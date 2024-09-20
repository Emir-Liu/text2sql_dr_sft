"""eval the memory usage of model"""

from transformers import AutoModelForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model_path = '/home/ymLiu/model/CodeLlama-7b-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_path)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)