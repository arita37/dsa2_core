# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

  python salary_regression.py  train
  python salary_regression.py  check
  python salary_regression.py  predict


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
config_default  = 'salary_lightgbm'
n_sample     = 1000




####################################################################################
cols_input_type_1 = {
     "coly"   :   "salary"
    ,"colid"  :   "jobId"
    ,"colcat" :   [ "companyId", "jobType", "degree", "major", "industry" ]
    ,"colnum" :   ["yearsExperience", "milesFromMetropolis"]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : [ "jobType", "degree", "major", "industry", "yearsExperience", "milesFromMetropolis" ]
}





####### y normalization #############################################################   
def y_norm(y, inverse=True, mode='boxcox'):
    ## Normalize the input/output
    if mode == 'boxcox':
        width0 = 53.0  # 0,1 factor
        k1 = 0.6145279599674994  # Optimal boxCox lambda for y
        if inverse:
                y2 = y * width0
                y2 = ((y2 * k1) + 1) ** (1 / k1)
                return y2
        else:
                y1 = (y ** k1 - 1) / k1
                y1 = y1 / width0
                return y1

    if mode == 'norm':
        m0, width0 = 0.0, 350.0  ## Min, Max
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
def salary_lightgbm(path_model_out="") :
    """
        Huber Loss includes L1  regurarlization
        We test different features combinaison, default params is optimal
    """
    data_name     = "salary"
    model_class   = 'LGBMRegressor'
    n_sample      = 10**5

    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')


    model_dict = {'model_pars':
        {'model_class': model_class
        ,'model_path':  path_model_out
        ,'model_pars':  {'objective': 'huber',


        }  # default
        ,'post_process_fun': copy.deepcopy( post_process_fun)
        ,'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },


    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                                    },

    'data_pars': {
            'cols_input_type' : cols_input_type_1

            # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
            ,'cols_model_group': [ 'colnum', 'colcat_bin']

           ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data

    }}

    ################################################################################################
    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, os_get_function_name() )
    return model_dict


 
def salary_elasticnetcv(path_model_out=""):
    global model_name
    model_name        = 'ElasticNetCV'

    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'ElasticNetCV'
        , 'model_path': path_model_out
        , 'model_pars': {}  # default ones
        , 'post_process_fun': copy.deepcopy(post_process_fun)
        , 'pre_process_pars': {'y_norm_fun' : copy.deepcopy(pre_process_fun),

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },



    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                                    },
    'data_pars': {
            'cols_input_type' : cols_input_type_1

         ,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]

         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
    }}


    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, model_class=os_get_function_name() )
    return model_dict





def salary_bayesian_pyro(path_model_out="") :
    global model_name
    model_name        = 'model_bayesian_pyro'
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'model_bayesian_pyro'
        , 'model_path': path_model_out
        , 'model_pars': {'input_width': 112, }  # default
        , 'post_process_fun': post_process_fun

        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },

    'compute_pars': {'compute_pars': {'n_iter': 1200, 'learning_rate': 0.01}
                                 , 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                                    'explained_variance_score', 'r2_score', 'median_absolute_error']
                                 , 'max_size': 1000000
                                 , 'num_samples': 300
     },
    'data_pars':  {
            'cols_input_type' : cols_input_type_1


            ,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]


           ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                            }}

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, os_get_function_name() )
    return model_dict





def salary_glm( path_model_out="") :
    global model_name
    model_name        = 'TweedieRegressor'
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')



    model_dict = {'model_pars': {'model_class': 'TweedieRegressor'  # Ridge
        , 'model_path': path_model_out
        , 'model_pars': {'power': 0, 'link': 'identity'}  # default ones
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },



        'compute_pars': {'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                         'explained_variance_score',  'r2_score', 'median_absolute_error']
                                        },

    'data_pars': {
            'cols_input_type' : cols_input_type_1
            ,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                                }
    }

    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, model_name, data_name)
    return model_dict







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
from core_run import train




###################################################################################
######### Check data ##############################################################
def check():
   pass



####################################################################################
####### Inference ##################################################################
# predict(config='', nsample=10000)
from core_run import predict





###########################################################################################################
###########################################################################################################
"""
python  salary_regression.py  preprocess
python  salary_regression.py  train
python  salary_regression.py  check
python  salary_regression.py  predict
python  salary_regression.py  run_all
"""
if __name__ == "__main__":
        import fire
        fire.Fire()

