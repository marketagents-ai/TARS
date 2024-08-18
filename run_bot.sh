 # Run the Python script with the specified API
   python -m tars_bot.main --api openrouter

   # Check if the script ran successfully
   if [ $? -eq 0 ]; then
       echo "Bot started successfully."
   else
       echo "Error: Failed to start the bot."
       exit 1
   fi