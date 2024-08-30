import argparse
from bot_setup import setup_bot
from api_client import initialize_api_client
from config import TOKEN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord bot with selected API and model')
    parser.add_argument('--api', choices=['azure', 'ollama', 'openrouter', 'localai'], default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, help='Specify the model to use. If not provided, defaults will be used based on the API.')
    args = parser.parse_args()

    initialize_api_client(args)

    bot = setup_bot()
    bot.run(TOKEN)
