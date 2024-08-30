import os
import logging
import aiohttp
import json
from dotenv import load_dotenv
from colorama import Fore, Back, Style, init
from datetime import datetime

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Global variables
API_TYPE = None
API_BASE = None
API_VERSION = None
API_KEY = None
DEPLOYMENT_NAME = None
MODEL_NAME = None

def initialize_api_client(args):
    global API_TYPE, API_BASE, API_VERSION, API_KEY, DEPLOYMENT_NAME, MODEL_NAME
    
    API_TYPE = args.api
    
    if API_TYPE == 'azure':
        API_BASE = os.getenv('AZURE_API_BASE')
        API_VERSION = os.getenv('AZURE_API_VERSION')
        API_KEY = os.getenv('AZURE_API_KEY')
        DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')
    elif API_TYPE == 'ollama':
        API_BASE = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
        MODEL_NAME = args.model or os.getenv('OLLAMA_MODEL_NAME', 'hermes3')
    elif API_TYPE == 'openrouter':
        API_BASE = "https://openrouter.ai/api/v1"
        API_KEY = os.getenv('OPENROUTER_API_KEY')
        MODEL_NAME = args.model or os.getenv('OPENROUTER_MODEL_NAME', 'nousresearch/hermes-3-llama-3.1-405b')
    elif API_TYPE == 'localai':
        API_BASE = os.getenv('LOCALAI_API_BASE', 'https://demo.localai.io/v1')
        MODEL_NAME = args.model or os.getenv('LOCALAI_MODEL_NAME', 'Hermes-3-Llama-3.1-8B:vllm')
    else:
        raise ValueError(f"Unsupported API type: {API_TYPE}")

    logging.info(f"Initialized API client with {API_TYPE}")

def log_to_jsonl(data):
    with open('api_calls.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

async def call_api(prompt, context="", system_prompt="", conversation_id=None):
    print(f"{Fore.YELLOW}System Prompt: {system_prompt}")
    print(f"{Fore.CYAN}User Input: {prompt}")

    if API_TYPE == 'azure':
        response = await call_azure_api(prompt, context, system_prompt)
    elif API_TYPE == 'ollama':
        response = await call_ollama_api(prompt, context, system_prompt)
    elif API_TYPE == 'openrouter':
        response = await call_openrouter_api(prompt, context, system_prompt)
    elif API_TYPE == 'localai':
        response = await call_localai_api(prompt, context, system_prompt)
    else:
        raise ValueError(f"Unsupported API type: {API_TYPE}")

    print(f"{Fore.GREEN}AI Output: {response}")

    # Log the API call
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "api_type": API_TYPE,
        "system_prompt": system_prompt,
        "context": context,
        "user_input": prompt,
        "ai_output": response
    }
    log_to_jsonl(log_data)

    return response

async def call_azure_api(prompt, context, system_prompt):
    url = f"{API_BASE}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"Azure API call failed with status {response.status}: {await response.text()}")

async def call_ollama_api(prompt, context, system_prompt):
    url = f"{API_BASE}/api/generate"
    headers = {"Content-Type": "application/json"}
    
    full_prompt = f"{system_prompt}\n\nContext: {context}\n\nHuman: {prompt}\nAI:"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['response'].strip()
            else:
                raise Exception(f"Ollama API call failed with status {response.status}: {await response.text()}")

async def call_openrouter_api(prompt, context, system_prompt):
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://your-app-domain.com", # Replace with your actual domain
        "X-Title": "Your App Name" # Replace with your app name
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"OpenRouter API call failed with status {response.status}: {await response.text()}")

async def call_localai_api(prompt, context, system_prompt):
    url = f"{API_BASE}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"LocalAI API call failed with status {response.status}: {await response.text()}")
