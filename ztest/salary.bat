


RMDIR /S /Q  data/output/salary/ >nul 2>&1

MKDIR  data\output\salary  >nul 2>&1


del  zlog/log_salary_prepro.txt >nul 2>&1
del  zlog/log_salary_train.txt  >nul 2>&1
del  zlog/log_salary_predict.txt >nul 2>&1


  python salary_regression.py  preprocess  --nsample 2000   > zlog/log_salary_prepro.txt 2>&1
  python salary_regression.py  train    --nsample 10000     > zlog/log_salary_train.txt 2>&1
  python salary_regression.py  predict  --nsample 5000     > zlog/log_salary_predict.txt 2>&1




