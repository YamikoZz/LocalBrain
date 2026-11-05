@echo off
setlocal
pushd "%~dp0"
if not exist "workspace\run-instruction\latest.pt" (
    echo Trained model not found. See README.md.
    pause
    popd
    exit /b 1
)
start "LocalBrain Learned Model" ".venv\Scripts\pythonw.exe" -m idlelib -t "LocalBrain Learned Model" -c "import sys; sys.argv=['localbrain','chat','--model-only','--run','workspace/run-instruction']; from localbrain.cli import main; main()"
popd
endlocal
