# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
To test encoding

"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')


###### Path ########################################################################
from source import util_feature
config_file  = os.path.basename(__file__)  ### name of file which contains data configuration

print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

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
    m['path_data_preprocess'] = root + f'/data/input/{data_name}/train/'

    ##### run_Train  ONLY
    m['path_data_train']   = root + f'/data/input/{data_name}/train/'
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
##### Params########################################################################
config_default   = 'titanic1'          ### name of function which contains data configuration


cols_input_type_2 = {
     "coly"   :   "Survived"
    ,"colid"  :   "PassengerId"
    ,"colcat" :   ["Sex", "Embarked" ]
    ,"colnum" :   ["Pclass", "Age","SibSp", "Parch","Fare"]
    ,"coltext" :  ["Name", "Ticket"]
    ,"coldate" :  []
    ,"colcross" : [ "Name", "Sex", "Ticket","Embarked","Pclass", "Age","SibSp", "Parch","Fare" ]

    ,'colgen'  : [  'Survived', "Pclass", "Age","SibSp", "Parch","Fare" ]
}


####################################################################################
def titanic1(path_model_out="") :
    """
       Contains all needed informations for Light GBM Classifier model,
       used for titanic classification task
    """
    config_name  = os_get_function_name()
    data_name    = "titanic"         ### in data/input/
    model_class  = 'LGBMClassifier'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 500

    def post_process_fun(y):
        return  int(y)

    def pre_process_fun(y):
        return  int(y)


    model_dict = {'model_pars': {
    ### LightGBM API model   #######################################
     'model_class': model_class
    ,'model_pars' : {'objective': 'binary', 'n_estimators':10,
                    }

    , 'post_process_fun' : post_process_fun
    , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


    ### Pipeline for data processing ##############################
    'pipe_list': [
        {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
        {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
        # {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
        {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        # {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
        # {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'},


        {'uri': 'source/preprocessors.py::pd_colcat_minhash',       'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_minhash',     'type': ''             },


        # {'uri': 'source/preprocessors.py::pd_coltext_universal_google',   'pars': {}, 'cols_family': 'coltext',     'cols_out': 'coltext_universal_google',     'type': ''    },


        {'uri': 'source/preprocessors.py::pd_col_genetic_transform',       'pars': {  ## 'pars_genetic' : {}
                                                                                   },
                'cols_family': 'colgen',     'cols_out': 'col_genetic',     'type': 'add_coly'             },


        {'uri': 'source/preprocessors.py::pd_colnum_quantile_norm',       'pars': {'colsparse' :  [] },
         'cols_family': 'colnum',     'cols_out': 'colnum_quantile_norm',     'type': ''             },


    ],
           }
    },

  'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                  },

  'data_pars': { 'n_sample' : n_sample,
      'cols_input_type' : cols_input_type_2,
      ### family of columns for MODEL  #########################################################
      #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
      #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
      #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
      #  'coldate',
      #  'coltext',
      'cols_model_group': [ 'colnum',  ### should be optional 'colcat'
          
                            'colcat_bin',
                            # 'colcat_bin',
                            # 'colnum_onehot',

                            #'colcat_minhash',
                            # 'colcat_onehot',
                            # 'coltext_universal_google'


                            'colcat_minhash',

                            'col_genetic',

                            'colnum_quantile_norm'


                          ]

      ### Filter data rows   ##################################################################
     ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name )
    return model_dict



def get_test_data(name='boston'):
    import pandas as pd
    if name == 'boston' :
        from sklearn.datasets import load_boston
        d = load_boston(return_X_y=False)
        col = d['feature_names']
        df = pd.DataFrame( d['data'], columns=col)
    return df, col


def check1():
    #### python core_test_encoder.py check1
    #############################################################
    from source.preprocessors import pd_col_genetic_transform  as pd_prepro
    pars = { 'path_pipeline_export' : ''

    }


    ############################################################
    for name in  ['boston'] :
        df, col = get_test_data(name)
        dfnew, col_pars = pd_prepro(df, col, pars)
        print(pd_prepro, name)
        print(dfnew[col].head(3).T,  col)
        print(dfnew.head(3).T,  col_pars)









###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess



##################################################################################
########## Train #################################################################
from core_run import train



####################################################################################
####### Inference ##################################################################
# predict(config='', nsample=10000)
from core_run import predict




###########################################################################################################
###########################################################################################################
"""
python  core_test_encoder.py  data_profile
python  core_test_encoder.py  preprocess  --nsample 100
python  core_test_encoder.py  train       --nsample 200
python  core_test_encoder.py  predict


"""
if __name__ == "__main__":
    import fire
    fire.Fire()
    

