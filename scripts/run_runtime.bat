@echo off
REM Run the Clash Royale bot runtime from the repository root (Windows).
cd /d "%~dp0.."
python "%~dp0runtime\\run_runtime.py" %*
exit /b %ERRORLEVEL%
