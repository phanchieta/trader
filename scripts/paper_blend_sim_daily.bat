@echo off
REM Agendador de Tarefas — SIMULADOR do blend (roda junto do campeao real, sem MT5,
REM sem enviar ordem). Marca a fechamento, credita CDI, aplica custo de giro.
REM Precisa dos parquets do Yahoo atualizados: rode antes pull_factor_universe.py,
REM pull_cdi.py e pull_futures_proxy.py (ou deixe-os no cron). Loga em
REM data\paper_blend_sim.log. Comparar com: python scripts\compare_paper.py
setlocal
cd /d "%~dp0\.."
set PY=%USERPROFILE%\anaconda3\envs\trade\python.exe
echo ==== %DATE% %TIME% ==== >> data\paper_blend_sim.log
"%PY%" scripts\pull_factor_universe.py >> data\paper_blend_sim.log 2>&1
"%PY%" scripts\pull_cdi.py             >> data\paper_blend_sim.log 2>&1
"%PY%" scripts\paper_blend_sim.py      >> data\paper_blend_sim.log 2>&1
endlocal
