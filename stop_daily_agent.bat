@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -m agent.main --stop-scheduler
) else (
    python -m agent.main --stop-scheduler
)

endlocal
