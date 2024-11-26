import os
import logging
import aiohttp
import json
from dotenv import load_dotenv
from colorama import Fore, Back, Style, init
from datetime import datetime
import openai
import anthropic
import base64
from io import BytesIO
from PIL import Image
import mimetypes

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
    
    if API_TYPE == 'ollama':
        API_BASE = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
        MODEL_NAME = args.model or os.getenv('OLLAMA_MODEL_NAME', 'llama3.2-vision')
    elif API_TYPE == 'openai':
        API_KEY = os.getenv('OPENAI_API_KEY')
        MODEL_NAME = args.model or os.getenv('OPENAI_MODEL_NAME', 'chatgpt-4o-latest')
        openai.api_key = API_KEY
    elif API_TYPE == 'anthropic':
        API_KEY = os.getenv('ANTHROPIC_API_KEY')
        MODEL_NAME = args.model or os.getenv('ANTHROPIC_MODEL_NAME', 'claude-3-5-sonnet-20241022')
    elif API_TYPE == 'vllm':
        API_BASE = os.getenv('VLLM_API_BASE', 'http://62.169.159.179:8000')
        MODEL_NAME = args.model or os.getenv('VLLM_MODEL_NAME', 'NousResearch/Hermes-3-Llama-3.1-70B')
    else:
        raise ValueError(f"Unsupported API type: {API_TYPE}")

    logging.info(f"Initialized API client with {API_TYPE}")

def log_to_jsonl(data):
    with open('api_calls.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

def encode_image(image_path):
    """Encode image to base64 with proper resizing if needed"""
    with Image.open(image_path) as img:
        # Resize if needed (max dimension 1568 for compatibility)
        max_dim = 1568
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prepare_image_content(prompt: str, image_paths: list, api_type: str) -> dict:
    """Prepare image content based on API requirements"""
    if not image_paths:
        return prompt

    base64_images = [encode_image(path) for path in image_paths]
    
    if api_type == 'anthropic':
        content = [{"type": "text", "text": prompt}]
        for img_data in base64_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_data
                }
            })
        return content
        
    elif api_type == 'openai':
        content = [{"type": "text", "text": prompt}]
        for img_data in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_data}"
                }
            })
        return content

    elif api_type == 'ollama':
        if len(base64_images) > 1:
            logging.warning("Ollama only supports one image per request. Using first image.")
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[0]}"
                }
            }
        ]
    
    else:
        raise ValueError(f"Unsupported API type for image handling: {api_type}")

async def call_api(prompt, context="", system_prompt="", conversation_id=None, temperature=0.7, image_paths=None):
    is_image = bool(image_paths)
    
    print(f"{Fore.YELLOW}System Prompt: {system_prompt}")
    print(f"{Fore.CYAN}User Input: {f'[Image] {prompt}' if is_image else prompt}")

    try:
        if is_image:
            formatted_content = prepare_image_content(prompt, image_paths, API_TYPE)
        else:
            formatted_content = prompt

        if API_TYPE == 'ollama':
            response = await call_ollama_api(formatted_content, context, system_prompt, temperature, is_image)
        elif API_TYPE == 'openai':
            response = await call_openai_api(formatted_content, context, system_prompt, temperature, is_image)
        elif API_TYPE == 'anthropic':
            response = await call_anthropic_api(formatted_content, context, system_prompt, temperature, is_image)
        elif API_TYPE == 'vllm':
            response = await call_vllm_api(formatted_content, context, system_prompt, temperature)
        else:
            raise ValueError(f"Unsupported API type: {API_TYPE}")

        print(f"{Fore.GREEN}AI Output: {response}")

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "api_type": API_TYPE,
            "system_prompt": system_prompt,
            "context": context,
            "user_input": f"[Image] {prompt}" if is_image else prompt,
            "ai_output": response,
            "is_image": is_image,
            "num_images": len(image_paths) if image_paths else 0
        }
        log_to_jsonl(log_data)

        return response
    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        raise

async def call_vllm_api(content, context, system_prompt, temperature):
    """Call vLLM API endpoint"""
    async with aiohttp.ClientSession() as session:
        # Combine prompts if present
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n"
        if context:
            full_prompt += f"{context}\n"
        full_prompt += content

        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "max_tokens": 4096,  # Default value, could be made configurable
            "temperature": 0.5 + temperature
        }
        
        try:
            async with session.post(
                f"{API_BASE}/v1/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"vLLM API returned status {response.status}: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["text"].strip()
        except aiohttp.ClientError as e:
            error_message = f"vLLM API request failed: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

async def call_ollama_api(content, context, system_prompt, temperature, is_image):
    client = openai.AsyncOpenAI(
        base_url=f"{API_BASE}/v1/",
        api_key="ollama"  # Required but ignored
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": content})
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=16384,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Ollama API error: {str(e)}")
        raise Exception(f"Ollama API call failed: {str(e)}")

async def call_openai_api(content, context, system_prompt, temperature, is_image):
    client = openai.AsyncOpenAI(api_key=API_KEY)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": content})
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=16384,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        raise Exception(f"OpenAI API call failed: {str(e)}")

async def call_anthropic_api(content, context, system_prompt, temperature, is_image):
    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    
    try:
        response = await client.messages.create(
            model=MODEL_NAME,
            system=system_prompt,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=4096,
            temperature=temperature
        )
        return response.content[0].text.strip()
    except anthropic.APIError as e:
        error_message = f"Anthropic API error: {str(e)}"
        logging.error(error_message)
        raise Exception(error_message)
    except Exception as e:
        error_message = f"Unexpected error in Anthropic API call: {str(e)}"
        logging.error(error_message)
        raise Exception(error_message)

async def get_embeddings(text, model=None):
    """Get embeddings from various API providers.
    
    Args:
        text (str): Text to get embeddings for
        model (str, optional): Override default embedding model
        
    Returns:
        list: Vector embeddings
    """
    if API_TYPE == 'openai':
        client = openai.AsyncOpenAI(api_key=API_KEY)
        embed_model = model or "text-embedding-3-small"
        try:
            response = await client.embeddings.create(
                model=embed_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"OpenAI embeddings error: {str(e)}")
            raise Exception(f"OpenAI embeddings failed: {str(e)}")
            
    elif API_TYPE == 'ollama':
        embed_model = model or "nomic-embed-text"
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{API_BASE}/api/embeddings"
                async with session.post(
                    url,
                    json={
                        "model": embed_model,
                        "prompt": text
                    }
                ) as response:
                    result = await response.json()
                    return result["embedding"]
            except Exception as e:
                logging.error(f"Ollama embeddings error: {str(e)}")
                raise Exception(f"Ollama embeddings failed: {str(e)}")
    
    elif API_TYPE == 'vllm':
        embed_model = model or os.getenv('VLLM_EMBED_MODEL', 'jinaai/jina-embeddings-v2-base-en')
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    "http://38.128.232.35:8080/embed",
                    json={
                        "model": embed_model,
                        "inputs": text
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"vLLM embeddings API returned status {response.status}: {error_text}")
                    
                    result = await response.json()
                    logging.debug(f"VLLM API Response: {result}")
                    
                    if isinstance(result, list):
                        return result[0]
                    elif isinstance(result, dict) and "embeddings" in result:
                        return result["embeddings"][0]
                    else:
                        return result
                        
            except Exception as e:
                logging.error(f"vLLM embeddings error: {str(e)}")
                raise Exception(f"vLLM embeddings failed: {str(e)}")
    
    else:
        raise ValueError(f"Embeddings not supported for API type: {API_TYPE}")




if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description='Multi-API LLM Client')
    parser.add_argument('--api', required=True, choices=['ollama', 'openai', 'anthropic', 'vllm'],
                      help='API type to use')
    parser.add_argument('--model', help='Model name (optional)')
    # Add other arguments as needed