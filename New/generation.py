import openai
from zhipuai import ZhipuAI
from http import HTTPStatus
import dashscope
import math
import traceback
import replicate
from openai import AzureOpenAI, OpenAI
import json
from tqdm import tqdm
import concurrent.futures
import os, time
from tenacity import retry, wait_random_exponential, stop_after_attempt
import urllib3
import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

deepinfra_model_mapping = {'Llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
                           'Llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
                           'Llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
                           'Llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct'}


qwen_api_key = config.qwen_api_key
openai_api = config.openai_api
deepinfra_api = config.deepinfra_api
zhipu_api = config.zhipu_api

Emotion_File = ['EmoBench_EA.json', 'EmoBench_EU.json']
Personality_File = ['big_five.json', 'dark_traits.json']

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string, model):
    model_mapping = {'gpt-4': 'yuehuang-gpt-4', 'chatgpt': 'yuehuang-chatgpt'}
    client = AzureOpenAI(
        api_key=openai_api,
        api_version="2023-12-01-preview",
        azure_endpoint="https://yuehuang-15w.openai.azure.com/"
    )
    try:
        chat_completion = client.chat.completions.create(
            model=model_mapping[model],
            messages=[
                {"role": "user", "content": string}
            ],
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None

def qwen_res(string):
    dashscope.api_key=qwen_api_key
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': string}]
    try:
        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            print(response)
            return response['output']['choices'][0].message.content
        else:
            print(response)
            return None
    except:
        return None

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def deepinfra_api(string, model, temperature):
    client = OpenAI(api_key=deepinfra_api,
                    base_url='https://api.deepinfra.com/v1/openai')
    top_p = 1 if temperature <= 1e-5 else 0.9
    # temperature=0.0001 if temperature<=1e-5 else temperature
    chat_completion = client.chat.completions.create(
        model=deepinfra_model_mapping[model],
        messages=[{"role": "user", "content": string}],
        max_tokens=2500,
        temperature=temperature,
        top_p=top_p,
    )
    return chat_completion.choices[0].message.content

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def zhipu_api(string, model, temperature):
    client = ZhipuAI(api_key=zhipu_api)
    if temperature == 0:
        temperature = 0.01
    else:
        temperature = 0.99
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": string},
        ],
        temperature=temperature
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


