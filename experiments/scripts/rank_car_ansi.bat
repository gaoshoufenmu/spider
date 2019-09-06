REM cd /d %~dp0
echo %cd%
cd /d %cd%
call E:\miniconda\Scripts\activate.bat
REM --cfg experiments/cfgs/rank_car.yml
REM to rank all category, please use '%cd%\tools\rank_car.py'
%cd%\tools\rank_car.py --cat �γ� --grade �����ͳ� --energy ȼ��