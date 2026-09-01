@echo off
REM Wrapper para o Agendador de Tarefas do Windows.
REM Requer o MT5 (conta demo) aberto e logado. Loga em data\paper_trade.log.
setlocal
cd /d "%~dp0\.."
set PY=%USERPROFILE%\anaconda3\envs\trade\python.exe
echo ==== %DATE% %TIME% ==== >> data\paper_trade.log
"%PY%" scripts\paper_trade.py --send >> data\paper_trade.log 2>&1
endlocal
