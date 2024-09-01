import json
from datetime import datetime
import logging
from config import TEMP_DIR
import os
import re
import discord
from api_client import call_api  # Add this line

def log_to_jsonl(data):
    with open('bot_log.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

async def process_message(message, memory_index, prompt_formats, system_prompts, cache_manager, github_repo):
    user_id = str(message.author.id)
    user_name = message.author.name
    
    # Check if the message is a command (starts with '!')
    is_command = message.content.startswith('!')
    
    if is_command:
        content = message.content.split(maxsplit=1)[1] if len(message.content.split()) > 1 else ""
    else:
        content = message.content.strip()
    
    logging.info(f"Received message from {user_name} (ID: {user_id}): {content}")

    try:
        relevant_memories = memory_index.search(content)
        conversation_history = cache_manager.get_conversation_history(user_id)
        
        context = f"Current channel: {message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n"
                
        if conversation_history:
            context += f"Previous conversation history with {user_name} (User ID: {user_id}):\n"
            for i, msg in enumerate(reversed(conversation_history[-5:]), 1):
                truncated_user_message = truncate_middle(msg['user_message'], max_length=100)
                truncated_ai_response = truncate_middle(msg['ai_response'], max_length=100)
                context += f"Interaction {i}:\n{msg['user_name']}: {truncated_user_message}\nAI: {truncated_ai_response}\n\n"
        else:
            context += f"This is the first interaction with {user_name} (User ID: {user_id}).\n"
        
        if relevant_memories:
            context += "Relevant memories:\n"
            for memory, score in relevant_memories:
                context += f"[Relevance: {score:.2f}] {memory}\n"
            context += "\n"
        
        prompt = prompt_formats['chat_with_memory'].format(
            context=context,
            user_name=user_name,
            user_message=content
        )

        system_prompt = system_prompts['default_chat']

        # Call the API only once
        response_content = await call_api(prompt, context=context, system_prompt=system_prompt)

        # Send the response
        await send_long_message(message.channel, response_content)

        memory_text = f"User {user_name} in {message.channel.name if hasattr(message.channel, 'name') else 'DM'}: {content}\nAI: {response_content}"
        memory_index.add_memory(user_id, memory_text)

        cache_manager.append_to_conversation(user_id, {
            'user_name': user_name,
            'user_message': content,
            'ai_response': response_content
        })

        log_to_jsonl({
            'event': 'chat_interaction',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'user_message': content,
            'ai_response': response_content
        })

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await message.channel.send(error_message)
        logging.error(f"Error in message processing for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({
            'event': 'chat_error',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'error': str(e)
        })

async def process_file(ctx, file_content, filename, memory_index, prompt_formats, system_prompts):
    user_id = str(ctx.author.id)
    user_name = ctx.author.name

    logging.info(f"Processing file '{filename}' from {user_name} (ID: {user_id})")

    try:
        file_prompt = prompt_formats['analyze_file'].format(
            filename=filename,
            file_content=file_content[:]  # Limit content to first 1000 characters
        )

        system_prompt = system_prompts['file_analysis']

        response_content = await call_api(file_prompt, context="", system_prompt=system_prompt)

        await send_long_message(ctx.channel, response_content)
        logging.info(f"Sent file analysis response to {user_name} (ID: {user_id}): {response_content[:100]}...")

        memory_index.add_memory(user_id, f"Analyzed file '{filename}' for User {user_name}. Analysis: {response_content}")

        log_to_jsonl({
            'event': 'file_analysis',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'filename': filename,
            'ai_response': response_content
        })

    except Exception as e:
        error_message = f"An error occurred while analyzing the file: {str(e)}"
        await ctx.channel.send(error_message)
        logging.error(f"Error in file analysis for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({
            'event': 'file_analysis_error',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'filename': filename,
            'error': str(e)
        })

def simple_sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text)

async def send_long_message(channel, message):
    sentences = simple_sent_tokenize(message)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > 1900:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    for chunk in chunks:
        await channel.send(chunk)

def truncate_middle(text, max_length=100):
    if len(text) <= max_length:
        return text
    side_length = (max_length - 3) // 2
    end_length = side_length + (max_length - 3) % 2
    return f"{text[:side_length]}...{text[-end_length:]}"

def create_markdown_file(filename, content):
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', filename)
    safe_filename = safe_filename[:50]  # Limit filename length
    md_filename = os.path.join(TEMP_DIR, f"{safe_filename}.md")
    
    with open(md_filename, 'w', encoding='utf-8') as md_file:
        md_file.write(content)
    
    return md_filename

async def send_markdown_file(ctx, filename, content):
    md_filename = create_markdown_file(filename, content)
    await ctx.send(file=discord.File(md_filename))
    os.remove(md_filename)