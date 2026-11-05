@echo off
rem Open an interactive window to try the code model (runs on CPU so training keeps the GPU).
pushd "%~dp0"
set RUN=workspace\run-english-sft
if not exist "%RUN%\latest.pt" set RUN=workspace\run-code-pretrain
start "LocalBrain Playground" cmd /k ".venv\Scripts\python.exe -m localbrain playground --run %RUN% --device cpu"
popd
