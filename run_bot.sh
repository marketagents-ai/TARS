#!/bin/bash

# Add the project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Python script with the specified API
python tars_bot/main.py --api openrouter

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Bot started successfully."
else
    echo "Error: Failed to start the bot."
    exit 1
fi