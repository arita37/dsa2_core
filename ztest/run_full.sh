

pwd
which python
ls .

# python outlier_predict.py  preprocess  ;
# python outlier_predict.py  train    --nsample 1000     ;
# python outlier_predict.py  predict  --nsample 1000   ;

# python multi_classifier.py  train    --nsample 10000   ;


python salary_regression.py  train   --nsample 1000
python airbnb_regression.py  predict  --nsample 1000

python cardif_regression.py  train   --nsample 1000


python airbnb_regression.py  train   --nsample 20000
python airbnb_regression.py  predict  --nsample 5000






python income_classifier.py  train    --nsample 1000   ;
python income_classifier.py  predict  --nsample 1000   ;




