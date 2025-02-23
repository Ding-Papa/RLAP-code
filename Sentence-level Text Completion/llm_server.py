from openai import OpenAI

def llama3_8B_api(query):

    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your api port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=10
    )
    try:
        chat_response = client.chat.completions.create(
        model="your model path of Llama-3-8B",
        messages=info,
        stop=["<|end_of_text|>"],
        temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return 'None'
    
def qwen25_14B_api(query):
    info = [{"role": "user", "content": query}]
    openai_api_key = "EMPTY"
    openai_api_base = "your api port"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=60
    )
    try:
        chat_response = client.chat.completions.create(
        model="your model path of Qwen2.5-14B-Instruct",
        messages=info,
        stop=["<|im_end|>"],
        temperature=0,
        )
        return chat_response.choices[0].message.content
    except:
        return 'None'