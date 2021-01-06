# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titatinic dataet
All in one file config
!  python jane_classifier.py  train
!  python jane_classifier.py  check
!  python jane_classifier.py  predict
"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys
import pandas as pd



###### Path ########################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)



###################################################################################
from source import util_feature




Metric = Perf% / Volatiliy

Perf = sum( weight * Perf * Action )

### Prediction 






####################################################################################
data_name    = "jane"     ### in data/input/
n_sample     = 200
dir0 =     dir_data   + f'/input/{data_name}/'
print(dir0)



df = pd.read_csv( dir0 + "/raw/train.zip" )


df.to_parquet( dir0 + "/raw/train.parquet"  )



df0 = df[df.date < 5 ]


df0.to_parquet( dir0 + "/raw/train_small.parquet"  )





df0 = df[df.date < 200 ]


df[ (df.date < 200) & (df.date >= 00)  ].to_parquet(dir0 + "/train01/train.parquet"  )

df[ (df.date < 350) & (df.date >= 200)  ].to_parquet(dir0 + "/train02/train.parquet"  )


df[ (df.date < 500 ) & (df.date >= 350)  ].to_parquet(dir0 + "/train03/train.parquet"  )



print(dict(df.dtypes))


print(list(df.columns))
















###########################################################################################################
###########################################################################################################
"""
python  jane_classifier.py  data_profile
python  jane_classifier.py  preprocess
python  jane_classifier.py  train
python  jane_classifier.py  check
python  jane_classifier.py  predict
python  jane_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()
    
