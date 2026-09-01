@echo off
REM Wrapper para o Agendador de Tarefas — executor do BLEND (campeao + fatores).
REM Requer o MT5 (conta demo) aberto e logado, e os parquets do universo de
REM fatores atualizados (rode pull_factor_universe.py + pull_futures_proxy.py
REM periodicamente). Loga em data\paper_blend.log.
setlocal
cd /d "%~dp0\.."
set PY=%USERPROFILE%\anaconda3\envs\trade\python.exe
echo ==== %DATE% %TIME% ==== >> data\paper_blend.log
"%PY%" scripts\paper_blend.py --send >> data\paper_blend.log 2>&1
endlocal
