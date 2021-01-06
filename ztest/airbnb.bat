


RMDIR /S /Q  data/output/airbnb/ >nul 2>&1

MKDIR  data\output\airbnb  >nul 2>&1


del zlog/log_airbnb_prepro.txt >nul 2>&1

del  zlog/log_airbnb_train.txt  >nul 2>&1

del  zlog/log_airbnb_predict.txt >nul 2>&1


  # python airbnb_regression.py   preprocess    > zlog/log_airbnb_prepro.txt 2>&1
  python3 airbnb_regression.py   train    > zlog/log_airbnb_train.txt 2>&1
  python3 airbnb_regression.py   predict    > zlog/log_airbnb_predict.txt 2>&1
