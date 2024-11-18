@echo off
set PYTHONPATH=%PYTHONPATH%;.
uvicorn gui.app.main:app --reload