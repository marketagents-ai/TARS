import json
import tiktoken
import os
from pathlib import Path

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def create_new_file(base_path: str, file_number: int) -> tuple:
    """Creates a new markdown file with incrementing number."""
    output_file = f"{base_path}_{file_number}.md"
    return open(output_file, 'w', encoding='utf-8'), output_file

def jsonl_to_markdown(input_file: str, output_base_path: str, max_tokens: int = 500000):
    # Initialize variables
    current_tokens = 0
    file_number = 1
    current_file = None
    files_created = []

    # Get base path without extension
    output_base = str(Path(output_base_path).with_suffix(''))
    
    try:
        # Open the first file
        current_file, current_filename = create_new_file(output_base, file_number)
        files_created.append(current_filename)

        # Read the JSONL file line by line
        with open(input_file, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract user input and AI output
                    user_input = data.get('user_input', '')
                    ai_output = data.get('ai_output', '')
                    
                    # Create formatted content
                    formatted_content = f"Q: {user_input}\n\nA: {ai_output}\n\n"
                    
                    # Count tokens in the formatted content
                    tokens_in_content = num_tokens_from_string(formatted_content)
                    
                    # Check if adding this content would exceed the token limit
                    if current_tokens + tokens_in_content > max_tokens:
                        # Close current file and create new one
                        current_file.close()
                        file_number += 1
                        current_file, current_filename = create_new_file(output_base, file_number)
                        files_created.append(current_filename)
                        current_tokens = 0
                    
                    # Write content to current file
                    current_file.write(formatted_content)
                    current_tokens += tokens_in_content
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                except Exception as e:
                    print(f"Error processing line: {e}")
                    
    finally:
        if current_file:
            current_file.close()
    
    return files_created

if __name__ == "__main__":
    input_file = "api_calls.jsonl"  # Replace with your input file path
    output_file = "output.md"   # Replace with your desired output file path
    
    created_files = jsonl_to_markdown(input_file, output_file)
    print("Conversion complete. Files created:")
    for file in created_files:
        print(f"- {file}")