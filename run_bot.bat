@echo off
setlocal EnableDelayedExpansion

:menu
cls
echo Discord Bot API Selection
echo ----------------------
echo 1. Ollama (Local)
echo 2. OpenAI
echo 3. Anthropic
echo 4. vLLM (Local)
echo ----------------------
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    cls
    echo Selected: Ollama
    python agent/discord_bot.py --api ollama
    goto end
)

if "%choice%"=="2" (
    cls
    echo Selected: OpenAI
    python agent/discord_bot.py --api openai
    goto end
)

if "%choice%"=="3" (
    cls
    echo Selected: Anthropic
    python agent/discord_bot.py --api anthropic
    goto end
)

if "%choice%"=="4" (
    cls
    echo Selected: vLLM
    python agent/discord_bot.py --api vllm
    goto end
)

if "%choice%"=="5" (
    echo Exiting...
    goto end
)

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:end
endlocal 