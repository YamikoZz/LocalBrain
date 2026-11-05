@echo off
rem Show LocalBrain training progress without asking Claude.
pushd "%~dp0"
".venv\Scripts\python.exe" -c "import json,time,pathlib;p=pathlib.Path('workspace/run-code-pretrain/metrics.jsonl');rows=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else [];r=rows[-1] if rows else None;print('No metrics yet - training is starting.' if not r else 'step %%d/20000 (%%.1f%%%%)  val_loss %%.3f  elapsed %%.0f min  ~%%.0f min left' %% (r['step'],r['step']/200,r['val_loss'],r['elapsed_seconds']/60,(20000-r['step'])*r['elapsed_seconds']/max(1,r['step'])/60))"
echo.
echo Last log lines:
powershell -NoProfile -Command "Get-Content workspace\run-code-pretrain.log -Tail 3"
popd
pause
