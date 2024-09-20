"""offline batched inference on a dataset
"""
from vllm import LLM, SamplingParams

prompts = [
    "hello",
    "hi",
    "hi, what can you do"
]

sampling_params = SamplingParams(
    temperature = 0,
    top_p = 0.95
)

model_ckpt = '/home/ymLiu/model/CodeLlama-7b-Instruct'

llm = LLM(model_ckpt)

outputs = llm.generate(
    prompts,
    sampling_params
)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'prompt:{prompt},\ngenerated_text:{generated_text}')


