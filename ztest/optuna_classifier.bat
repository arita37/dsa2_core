



RMDIR /S /Q  data/output/titanic/ >nul 2>&1

MKDIR  data\output\titanic  >nul 2>&1


del zlog/log_optuna_cls_prepro.txt >nul 2>&1

del  zlog/log_optuna_cls_train.txt  >nul 2>&1

del  zlog/log_optuna_cls_predict.txt >nul 2>&1


  python optuna_classifier.py  preprocess    > zlog/log_optuna_cls_prepro.txt 2>&1
  python optuna_classifier.py  train    > zlog/log_optuna_cls_train.txt 2>&1
  python optuna_classifier.py  predict  > zlog/log_optuna_cls_predict.txt 2>&1


