# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

  cd fraud
  python test_classifier.py  data_profile  --path_data_train data/input/titanic/train/  --path_out zlog/
  python test_classifier.py  preprocess
  python test_classifier.py  train
  python test_classifier.py  check
  python test_classifier.py  predict


NameError: Module ['test_classifier.py', 'pd_myfun'] notfound, No module named 'test_classifier.py'; 'test_classifier' is not a package, tuple index out of range


ip,app,device,os,channel,click_time,attributed_time,is_attributed
83230,3,1,13,379,11/6/2017 14:32,,0
17357,3,1,19,379,11/6/2017 14:33,,0
35810,3,1,13,379,11/6/2017 14:34,,0
45745,14,1,13,478,11/6/2017 14:34,,0



"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

####################################################################################
###### Path ########################################################################
from source.util_feature import save

config_file  = os.path.basename(__file__)

print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

sys.path.append(root + "/bin/")

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name


def global_pars_update(model_dict,  data_name, config_name):
    m                      = {}
    m['config_path']       = root + f"/{config_file}"
    m['config_name']       = config_name

    ##### run_Preoprocess ONLY
    m['path_data_preprocess'] = root + f'/data/input/{data_name}/train_100k/'

    ##### run_Train  ONLY
    m['path_data_train']   = root + f'/data/input/{data_name}/train_100k/'
    m['path_data_test']    = root + f'/data/input/{data_name}/test/'
    #m['path_data_val']    = root + f'/data/input/{data_name}/test/'
    m['path_train_output']    = root + f'/data/output/{data_name}/{config_name}/'
    m['path_train_model']     = root + f'/data/output/{data_name}/{config_name}/model/'
    m['path_features_store']  = root + f'/data/output/{data_name}/{config_name}/features_store/'
    m['path_pipeline']        = root + f'/data/output/{data_name}/{config_name}/pipeline/'


    ##### Prediction
    m['path_pred_data']    = root + f'/data/input/{data_name}/test/'
    m['path_pred_pipeline']= root + f'/data/output/{data_name}/{config_name}/pipeline/'
    m['path_pred_model']   = root + f'/data/output/{data_name}/{config_name}/model/'
    m['path_pred_output']  = root + f'/data/output/{data_name}/pred_{config_name}/'


    #####  Generic
    m['n_sample']             = model_dict['data_pars'].get('n_sample', 5000)

    model_dict[ 'global_pars'] = m
    return model_dict



####################################################################################
config_default  = 'adfraud_lightgbm'   ### name of function which contains data configuration



####################################################################################
##### Params########################################################################
"""
ip                                uint32
app                               uint16
device                             uint8
os                                 uint8
channel                            uint8
is_attributed                      uint8
hour                               uint8
minute                             uint8
second                             uint8
day                                uint8
day_of_week                        uint8
day_section                      float64
n_ip_clicks                        uint8
n_app_clicks                       uint8
n_channels                         uint8
ip_app_count                       uint8
ip_app_os_count                    uint8
n_ip_os_day_hh                     uint8
n_ip_app_day_hh                    uint8
n_ip_app_os_day_hh                 uint8
n_ip_app_dev_os                    uint8
n_ip_dev_os                        uint8
ip_day_hour_count_channel          int64
ip_app_count_channel               int64
ip_app_os_count_channel            int64
ip_app_day_hour_count_channel      int64
ip_app_channel_mean_hour         float64
app_AvgViewPerDistinct_ip        float64
app_count_channel                  int64
channel_count_app                  int64
ip_nunique_channel                 int64
ip_nunique_app                     int64
ip_day_nunique_hour                int64
ip_app_nunique_os                  int64
ip_nunique_device                  int64
app_nunique_channel                int64
ip_device_os_nunique_app           int64
ip_device_os_cumcount_app          int64
ip_cumcount_app                    int64
ip_cumcount_os                     int64


"""
cols_input_type_1 = {
     "coly"   :   "is_attributed"     ### 
    ,"colid"  :   "Id"
    ,"colcat" :   [ "ip", "app", "device", "os", "channel", 'hour', 'day_of_week'  ]
    ,"colnum" :   [ 'n_ip_clicks', 'n_app_clicks', 'ip_app_channel_mean_hour'  ]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : []
}




####################################################################################
def  adfraud_lightgbm(path_model_out="") :
    """

    """
    config_name  = os_get_function_name()
    data_name    = "adfraud"         ### in data/input/
    model_class  = 'LGBMClassifier'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 1000

    def post_process_fun(y):   ### After prediction is done
        return  int(y)

    def pre_process_fun(y):    ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        ### LightGBM API model   #######################################
         'model_class': model_class
        ,'model_pars' : {'objective': 'binary',
                           'n_estimators': 10,
                           'learning_rate':0.001,
                           'boosting_type':'gbdt',     ### Model hyperparameters
                           'early_stopping_rounds': 5
                        }

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################


        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            # {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair',  'type': 'cross'},

            #### Example of Custom processor
            # {'uri': 'titanic_classifier.py::pd_colnum_quantile_norm',   'pars': {}, 'cols_family': 'colnum',   'cols_out': 'colnum_quantile_norm',  'type': '' },          
          
        ],
               }
        },

      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                        },

      'data_pars': { 'n_sample' : n_sample,
          'cols_input_type' : cols_input_type_1,
          ### family of columns for MODEL  #########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',
          'cols_model_group': [ 'colnum_bin',
                                'colcat_bin',
                                # 'coltext',
                                # 'coldate',
                                # 'colcross_pair',
                               
                               ### example of custom
                               # 'colnum_quantile_norm'
                              ]

          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name )
    return model_dict

# from adfraud import adfraud_lightgbm
# print( adfraud_lightgbm )



#####################################################################################
########## Profile data #############################################################
def data_profile(path_data_train="", path_model="", n_sample= 5000):
   from source.run_feature_profile import run_profile
   run_profile(path_data   = path_data_train,
               path_output = path_model + "/profile/",
               n_sample    = n_sample,
              )


###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess


##################################################################################
########## Train #################################################################
## def train(config_uri='titanic_classifier.py::titanic_lightgbm'):
from core_run import train



###################################################################################
######### Check data ##############################################################
def check():
   pass




####################################################################################
####### Inference ##################################################################
# def  predict(config='', nsample=10000)
from core_run import predict





###########################################################################################################
###########################################################################################################
"""
python   adfraud.py  data_profile
python   adfraud.py  preprocess  --nsample 100
python   adfraud.py  train       --nsample 200
python   adfraud_classifier.py  check
python   adfraud_classifier.py  predict
python   adfraud_classifier.py  run_all


"""
if __name__ == "__main__":

    import fire
    fire.Fire()
    


