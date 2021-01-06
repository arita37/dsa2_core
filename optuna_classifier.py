# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titanic dataset (along with optuna)
All in one file config
  python optuna_classifier.py  train    > zlog/log_optuna_cls_train.txt 2>&1
  python optuna_classifier.py  predict  > zlog/log_optuna_cls_predict.txt 2>&1


"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

####################################################################################
###### Path ########################################################################
from source import util_feature

config_file = os.path.basename(__file__)
print(config_file)

print(os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data = os.path.abspath(root + "/data/") + "/"
dir_data = dir_data.replace("\\", "/")
print(dir_data)


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name


def global_pars_update(model_dict, data_name, config_name):
    m = {}
    m['config_path'] = root + f"/{config_file}"
    m['config_name'] = config_name

    ##### run_Preoprocess ONLY
    m['path_data_preprocess'] = root + f'/data/input/{data_name}/train/'

    ##### run_Train  ONLY
    m['path_data_train'] = root + f'/data/input/{data_name}/train/'
    m['path_data_test'] = root + f'/data/input/{data_name}/test/'
    # m['path_data_val']    = root + f'/data/input/{data_name}/test/'
    m['path_train_output'] = root + f'/data/output/{data_name}/{config_name}/'
    m['path_train_model'] = root + f'/data/output/{data_name}/{config_name}/model/'
    m['path_features_store'] = root + f'/data/output/{data_name}/{config_name}/features_store/'
    m['path_pipeline'] = root + f'/data/output/{data_name}/{config_name}/pipeline/'

    ##### Prediction
    m['path_pred_data'] = root + f'/data/input/{data_name}/test/'
    m['path_pred_pipeline'] = root + f'/data/output/{data_name}/{config_name}/pipeline/'
    m['path_pred_model'] = root + f'/data/output/{data_name}/{config_name}/model/'
    m['path_pred_output'] = root + f'/data/output/{data_name}/pred_{config_name}/'

    #####  Generic
    m['n_sample'] = model_dict['data_pars'].get('n_sample', 5000)

    model_dict['global_pars'] = m
    return model_dict


####################################################################################
##### Params########################################################################
config_default = 'titanic_lightoptuna'  ### name of function which contains data configuration

# data_name    = "titanic"     ### in data/input/
cols_input_type_1 = {
    "coly": "Survived"
    , "colid": "PassengerId"
    , "colcat": ["Sex", "Embarked"]
    , "colnum": ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    , "coltext": []
    , "coldate": []
    , "colcross": ["Name", "Sex", "Ticket", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"]
}

cols_input_type_2 = {
    "coly": "Survived"
    , "colid": "PassengerId"
    , "colcat": ["Sex", "Embarked"]
    , "colnum": ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    , "coltext": ["Name", "Ticket"]
    , "coldate": []
    , "colcross": ["Name", "Sex", "Ticket", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"]
}


####################################################################################
def titanic_lightoptuna():
    """
       Contains all needed informations for Light GBM Classifier model,
       used for titanic classification task
    """
    config_name = os_get_function_name()
    data_name   = "titanic"  ### in data/input/
    model_class = 'LGBMModel_optuna'  ### ACTUAL Class name for model_sklearn.py
    n_sample    = 1000

    def post_process_fun(y):
        ### After prediction is done
        return int(y)

    def pre_process_fun(y):
        ### Before the prediction is done
        return int(y)

    model_dict = {'model_pars': {
    ### LightGBM API model   #######################################
    'model_file'  : 'model_optuna.py',  ###Optional one
    'model_class' :  model_class
    ,'model_pars':  {'objective': 'binary',
                     'n_estimators': 50,
                     'learning_rate': 0.001,
                     'boosting_type': 'gbdt',  ### Model hyperparameters
                     'early_stopping_rounds': 5
                     }


    ### After prediction  ##########################################
    , 'post_process_fun': post_process_fun

    ### Before training  ##########################################
    , 'pre_process_pars': {'y_norm_fun': pre_process_fun,

       ### Pipeline for data processing ##############################
       'pipe_list': [
           {'uri': 'source/preprocessors.py::pd_coly', 'pars': {}, 'cols_family': 'coly', 'cols_out': 'coly', 'type': 'coly'},
           {'uri': 'source/preprocessors.py::pd_colnum_bin', 'pars': {},  'cols_family': 'colnum', 'cols_out': 'colnum_bin', 'type': ''},
           # {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot', 'type': ''},

           {'uri': 'source/preprocessors.py::pd_colcat_bin', 'pars': {}, 'cols_family': 'colcat', 'cols_out': 'colcat_bin', 'type': ''},
           # {'uri': 'source/preprocessors.py::pd_colcat_to_onehot', 'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot', 'type': ''},
           #{'uri': 'source/preprocessors.py::pd_colcross', 'pars': {},'cols_family': 'colcross', 'cols_out': 'colcross_pair_onehot', 'type': 'cross'}
       ],
       }
    },

    #classoptuna.integration.lightgbm.LightGBMTuner(params: Dict[str, Any], train_set: lgb.Dataset,
    # num_boost_round: int = 1000, valid_sets: Optional[VALID_SET_TYPE] = None,
    # valid_names: Optional[Any] = None, fobj: Optional[Callable[[…], Any]] = None,
    # feval: Optional[Callable[[…], Any]] = None, feature_name: str = 'auto', categorical_feature: str = 'auto', early_stopping_rounds: Optional[int] = None, evals_result: Optional[Dict[Any, Any]] = None, verbose_eval: Union[bool, int, None] = True, learning_rates: Optional[List[float]] = None, keep_training_booster: bool = False, callbacks: Optional[List[Callable[[…], Any]]] = None, time_budget: Optional[int] = None, sample_size: Optional[int] = None, study: Optional[optuna.study.Study] = None, optuna_callbacks: Optional[List[Callable[[optuna.study.Study, optuna.trial._frozen.FrozenTrial], None]]] = None, model_dir: Optional[str] = None, verbosity: Optional[int] = None, show_progress_bar: bool = True)[source]
    'compute_pars': {'metric_list': ['accuracy_score', 'average_precision_score'],
                     'optuna_params': {
                         "early_stopping_rounds": 5,
                          'verbose_eval' :        100,
                           #  folds=KFold(n_splits=3)
                     },

                     'optuna_engine' : 'LightGBMTuner'   ###  LightGBMTuner', LightGBMTunerCV

                     },




    'data_pars': {'n_sample': n_sample,
                  'cols_input_type': cols_input_type_1,
                  ### family of columns for MODEL  #########################################################
                  #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
                  #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
                  #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
                  #  'coldate',
                  #  'coltext',
                  'cols_model_group': ['colnum_bin',
                                       'colcat_bin',
                                       # 'coltext',
                                       # 'coldate',
                                       # 'colcross_pair'
                                       ]

                  ### Filter data rows   ##################################################################
        , 'filter_pars': {'ymax': 2, 'ymin': -1}

                  }
    }

    ##### Filling Global parameters    ############################################################
    model_dict = global_pars_update(model_dict, data_name, config_name)
    return model_dict


#####################################################################################
########## Profile data #############################################################
def data_profile(path_data_train="", path_model="", n_sample=5000):
    from source.run_feature_profile import run_profile
    run_profile(path_data=path_data_train,
                path_output=path_model + "/profile/",
                n_sample=n_sample,
                )


###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess

"""
def preprocess(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_preprocess
    run_preprocess.run_preprocess(config_name   =  config_name,
                                  config_path   =  m['config_path'],
                                  n_sample      =  nsample if nsample is not None else m['n_sample'],

                                  ### Optonal
                                  mode          =  'run_preprocess')
"""

##################################################################################
########## Train #################################################################
from core_run import train

"""
def train(config=None, nsample=None):

    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_train
    run_train.run_train(config_name       =  config_name,
                        config_path       =  m['config_path'],
                        n_sample          =  nsample if nsample is not None else m['n_sample'],
                        )
"""


###################################################################################
######### Check data ##############################################################
def check():
    pass


####################################################################################
####### Inference ##################################################################
# predict(config='', nsample=10000)
from core_run import predict

"""
def predict(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']


    from source import run_inference
    run_inference.run_predict(config_name = config_name,
                              config_path = m['config_path'],
                              n_sample    = nsample if nsample is not None else m['n_sample'],

                              #### Optional
                              path_data   = m['path_pred_data'],
                              path_output = m['path_pred_output'],
                              model_dict  = None
                              )
"""

###########################################################################################################
###########################################################################################################
"""
python  optuna_classifier.py  data_profile
python  optuna_classifier.py  preprocess  --nsample 100
python  optuna_classifier.py  train       --nsample 200
python  optuna_classifier.py  check
python  optuna_classifier.py  predict


"""
if __name__ == "__main__":
    import fire

    fire.Fire()


