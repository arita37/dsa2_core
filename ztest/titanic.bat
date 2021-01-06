



RMDIR /S /Q  data/output/income/ >nul 2>&1

MKDIR  data\output\income  >nul 2>&1


del zlog/log_income_prepro.txt >nul 2>&1

del  zlog/log_income_train.txt  >nul 2>&1

del  zlog/log_income_predict.txt >nul 2>&1


  python income_classifier.py  preprocess    > zlog/log_income_prepro.txt 2>&1
  python income_classifier.py  train    > zlog/log_income_train.txt 2>&1
  python income_classifier.py  predict  > zlog/log_income_predict.txt 2>&1


