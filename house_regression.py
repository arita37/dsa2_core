# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titatinic dataet
All in one file config
  python house_regression.py  train    > zlog/log-house.txt 2>&1
!  python house_regression.py  check
!  python house_regression.py  predict


# 'pipe_list'  :
    'filter',
    'label',
    'dfnum_bin'
    'dfnum_hot'
    'dfcat_bin'
    'dfcat_hot'
    'dfcross_hot'




"""
import warnings
warnings.filterwarnings('ignore')
import os, sys, copy
############################################################################
from source import util_feature



####################################################################################################
###### Path ########################################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


def global_pars_update(model_dict,  data_name, config_name):
    global path_config_model, path_model, path_data_train, path_data_test, path_output_pred, n_sample,model_name
    model_name        = model_dict['model_pars']['model_class']
    path_config_model = root + f"/{config_file}"
    path_model        = f'data/output/{data_name}/a01_{model_name}/'
    path_data_train   = f'data/input/{data_name}/train/'
    path_data_test    = f'data/input/{data_name}/test/'
    path_output_pred  = f'/data/output/{data_name}/pred_a01_{config_name}/'

    model_dict[ 'global_pars'] = {}
    global_pars = [ 'model_class', 'model_class', 'config_path', 'path_model', 'path_data_train',
                   'path_data_test', 'path_output_pred', 'n_sample'
            ]
    for t in global_pars:
      model_dict['global_pars'][t] = globals()[t]
    return model_dict


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name



####################################################################################
config_file  = "house_regression.py"
data_name    = "house_price"


config_name  = 'house_price_lightgbm'
n_sample     = 10000
tag_job      = 'aa1'  ## to have a unique tag for the run



cols_input_type_2 = {
     "coly"   : "SalePrice"
    ,"colid"  : "Id"

    ,"colcat" : [ "MSSubClass", "MSZoning", "Street" ]

    ,"colnum" : [ "LotArea", "OverallQual", "OverallCond", 	]

    ,"coltext"  : []
    ,"coldate" : []   # ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
    ,"colcross" : []

}



cols_input_type_1 = {
     "coly"   : "SalePrice"
    ,"colid"  : "Id"

    ,"colcat" : [  "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
          "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

    ,"colnum" : [ "LotArea", "OverallQual", "OverallCond", "MasVnrArea",
          "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]

    ,"coltext"  : []
    ,"coldate" : []   # ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
    ,"colcross" : []

}




#####################################################################################
####### y normalization #############################################################
def y_norm(y, inverse=True, mode='boxcox'):
    ## Normalize the input/output
    if mode == 'boxcox':
        width0 = 53.0  # 0,1 factor
        k1 = 1.0  # Optimal boxCox lambda for y
        if inverse:
                y2 = y * width0
                y2 = ((y2 * k1) + 1) ** (1 / k1)
                return y2
        else:
                y1 = (y ** k1 - 1) / k1
                y1 = y1 / width0
                return y1

    if mode == 'norm':
        m0, width0 = 0.0, 100000.0  ## Min, Max
        if inverse:
                y1 = (y * width0 + m0)
                return y1

        else:
                y2 = (y - m0) / width0
                return y2
    else:
            return y



####################################################################################
##### Params########################################################################
def house_price_lightgbm(path_model_out="") :
    """
        Huber Loss includes L1  regurarlization
        We test different features combinaison, default params is optimal
    """
    data_name         = 'house_price'
    model_name        = 'LGBMRegressor'
    n_sample          = 20000


    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')


    model_dict = {'model_pars': {  'model_path'       : path_model_out

        , 'model_class': model_name   ### Actual Class Name
        , 'model_pars'       : {}  # default ones of the model name

        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' : copy.deepcopy(pre_process_fun),

            ### Pipeline for data processing.
            # 'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
           'pipe_list'  : [ 'filter', 'label',   'dfcat_bin'  ]

           }
                                                         },
    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                    },
    'data_pars': {
        'cols_input_type' : cols_input_type_1,

        # 'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]
        'cols_model_group': [ 'colnum', 'colcat_bin' ]


       ,'filter_pars': { 'ymax' : 1000000.0 ,'ymin' : 0.0 }   ### Filter data

    }}


    ################################################################################################
    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict







def house_price_elasticnetcv(path_model_out=""):
    model_name   = 'ElasticNetCV'
    config_name  = 'house_price_elasticnetcv'
    n_sample     = 1000


    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')


    model_dict = {'model_pars': {'model_class': 'ElasticNetCV'
        , 'model_path': path_model_out



        , 'model_pars': {}  # default ones
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

                        ### Pipeline for data processing.
                       # 'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
                       'pipe_list' : [ 'filter', 'label',   'dfcat_hot' ]
                                                     }
                                                         },
    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                    },

    'data_pars': {
        'cols_input_type' : cols_input_type_1,

        # 'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]
        'cols_model_group': [ 'colnum', 'colcat_onehot' ]

         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
    }}


    ################################################################################################
    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict




####################################################################################################
########## Init variable ###########################################################################
globals()[config_name]()



###################################################################################
########## Profile data #############################################################
def data_profile():
   from source.run_feature_profile import run_profile
   run_profile(path_data   = path_data_train,
               path_output = path_model + "/profile/",
               n_sample    = 5000,
              )



###################################################################################
########## Preprocess #############################################################
def preprocess():
    from source import run_preprocess_old
    run_preprocess_old.run_preprocess(model_name =  config_name,
                                      path_data         =  path_data_train,
                                      path_output       =  path_model,
                                      path_config_model =  path_config_model,
                                      n_sample          =  n_sample,
                                      mode              =  'run_preprocess')


############################################################################
########## Train ###########################################################
def train():
    from source import run_train
    run_train.run_train(config_name=  config_name,
                        path_data_train=  path_data_train,
                        path_output       =  path_model,
                        config_path=  path_config_model, n_sample = n_sample)


###################################################################################
######### Check model #############################################################
def check():
   pass


########################################################################################
####### Inference ######################################################################
def predict():
    from source import run_inference
    run_inference.run_predict(model_name,
                            path_model  = path_model,
                            path_data   = path_data_test,
                            path_output = path_output_pred,
                            n_sample    = n_sample)


def run_all():
    data_profile()
    preprocess()
    train()
    check()
    predict()



###########################################################################################################
###########################################################################################################
"""
python  house_regression.py  preprocess
python  house_regression.py  train
python  house_regression.py  check
python  house_regression.py  predict
python  house_regression.py  run_all
"""
if __name__ == "__main__":
        import fire
        fire.Fire()
