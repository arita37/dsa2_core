



RMDIR /S /Q  data/output/cardif/ >nul 2>&1

MKDIR  data\output\cardif  >nul 2>&1


del zlog/log_cardif_prepro.txt >nul 2>&1

del  zlog/log_cardif_train.txt  >nul 2>&1

del  zlog/log_cardif_predict.txt >nul 2>&1


  python cardif_classifier.py  preprocess  > zlog/log_cardif_prepro.txt 2>&1
  python cardif_classifier.py  train       > zlog/log_cardif_train.txt 2>&1
  python cardif_classifier.py  predict     > zlog/log_cardif_predict.txt 2>&1





