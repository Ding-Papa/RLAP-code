from openai import OpenAI
import os

def mistral_7B_api(query):

    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your_api_port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=5
    )
    try:
        chat_response = client.chat.completions.create(
        model="models/Mistral-7B-Instruct-v0.3",
        messages=info,
        stop=["<|im_end|>","</s>"],
        temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return 'None'

def qwen_14B_api(query):

    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your_api_port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=5
    )
    try:
        chat_response = client.chat.completions.create(
        model="Qwen/Qwen1.5-14B-Chat",
        messages=info,
        stop=["<|im_end|>"],
        temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return "None"

def qwen25_14B_api(query):
    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your_api_port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=5
    )
    try:
        chat_response = client.chat.completions.create(
        model="Qwen2.5-14B-Instruct/",
        messages=info,
        stop=["<|im_end|>"],
        temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return "None"