REM cd /d %~dp0
echo %cd%
cd /d %cd%
call E:\miniconda\Scripts\activate.bat
REM --cfg experiments/cfgs/rank_car.yml
%cd%\tools\rank_car.py --cat 轿车 --grade 紧凑型车 --energy 燃油
REM to rank all category, please use '%cd%\tools\rank_car.py'