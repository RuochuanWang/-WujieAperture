@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=D:\Anaconda\envs\test\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [WujieAperture] Python test environment not found: %PYTHON_EXE%
  pause
  exit /b 1
)
set "BOKEH_HOST=127.0.0.1"
set "BOKEH_PORT=7861"
echo [WujieAperture] Open http://127.0.0.1:7861 after the server starts.
"%PYTHON_EXE%" -m competition_app
if errorlevel 1 pause
