@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo LocalBrain Python environment not found. See README.md.
    pause
    popd
    exit /b 1
)
start "LocalBrain Thai Chat" ".venv\Scripts\pythonw.exe" -m idlelib -t "LocalBrain Thai Chat" -c "import sys; sys.argv=['localbrain','chat','--web']; from localbrain.cli import main; main()"
popd
endlocal
