# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Run template


python core_run.py data_profile --config outlier_predict.py::titanic_lightgbm

                                                                                     
python core_run.py preprocess --config outlier_predict.py::titanic_lightgbm


python core_run.py train --config outlier_predict.py::titanic_lightgbm


python core_run.py predict --config outlier_predict.py::titanic_lightgbm



"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')


####################################################################################
from source.util_feature import log

print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


 
def get_global_pars(config_uri=""):
  log("#### Model Params Dynamic loading  ##########################################")
  from source.util_feature import load_function_uri
  model_dict_fun = load_function_uri(uri_name=config_uri )

  #### Get dict + Update Global variables
  try :
     model_dict     = model_dict_fun()   ### params
  except :
     model_dict  = model_dict_fun

  return model_dict


def get_config_path(config=''):
    #### Get params where the file is imported  #####################
    path0 =  os.path.abspath( sys.modules['__main__'].__file__)
    print("file where imported", path0)

    config_default = get_global_pars( path0 + "::config_default")


    if len(config)  == 0 :
        config_uri  = path0  + "::" + config_default
        config_name = config_default

    elif "::" not in config :
        config_uri  = path0  + "::" + config
        config_name = config

    else :
        config_uri  = config
        config_name = config.split("::")[1]
    ##################################################################
    print("default: ", config_uri)
    return config_uri, config_name


#####################################################################################
########## Profile data #############################################################
def data_profile(config=''):
    """

    """
    config_uri, config_name = get_config_path(config)
    from source.run_feature_profile import run_profile
    mdict = get_global_pars( config_uri)
    m     = mdict['global_pars']
    log(mdict)

    run_profile(path_data   = m['path_data_train'],
               path_output  = m['path_model'] + "/profile/",  
               n_sample     = 5000,
              ) 


###################################################################################
########## Preprocess #############################################################
def preprocess(config='', nsample=None):
    """


    """
    config_uri, config_name = get_config_path(config)
    mdict = get_global_pars( config_uri)
    m     = mdict['global_pars']
    log(mdict)

    from source import run_preprocess
    run_preprocess.run_preprocess(config_name   =  config_name,
                                  config_path   =  m['config_path'],
                                  n_sample      =  nsample if nsample is not None else m['n_sample'],

                                  ### Optonal
                                  mode          =  'run_preprocess')


####################################################################################
########## Train ###################################################################
def train(config='', nsample=None):

    config_uri, config_name = get_config_path(config)

    mdict = get_global_pars(  config_uri)
    m     = mdict['global_pars']
    log(mdict)
    from source import run_train
    run_train.run_train(config_name       =  config_name,
                        config_path       =  m['config_path'],
                        n_sample          =  nsample if nsample is not None else m['n_sample'],
                        )


####################################################################################
######### Check model ##############################################################
def check(config='outlier_predict.py::titanic_lightgbm'):
    mdict = get_global_pars(config)
    m     = mdict['global_pars']
    log(mdict)
    pass




########################################################################################
####### Inference ######################################################################
def predict(config='', nsample=None):

    config_uri, config_name = get_config_path(config)

    mdict = get_global_pars( config_uri)
    m     = mdict['global_pars']
    log(mdict)


    from source import run_inference
    run_inference.run_predict(config_name = config_name,
                              config_path = m['config_path'],
                              n_sample    = nsample if nsample is not None else m['n_sample'],

                              #### Optional
                              path_data   = m['path_pred_data'],
                              path_output = m['path_pred_output'],
                              model_dict  = None
                              )


##########################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
    








