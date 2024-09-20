
from openai import OpenAI

openai_api_key = 'none'
openai_api_base = 'http://localhost:8000/v1'
client = OpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base,
)

completion = client.completions.create(
    model = '/home/ymLiu/model/chatglm3-6b/',
    prompt = 'hi'
)

print(f'completion res:{completion}')
