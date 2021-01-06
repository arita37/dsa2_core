# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
All in one file config
  python cardif_classifier.py  preprocess
  python cardif_classifier.py  train
  python cardif_classifier.py  predict
"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys

############################################################################
from source import util_feature


###### Path ################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


def global_pars_update(model_dict,  data_name, config_name):
    m                      = {}
    model_name             = model_dict['model_pars']['model_class']
    m['config_path'] = root + f"/{config_file}"
    m['config_name']       = config_name

    m['path_data_train']   = f'data/input/{data_name}/train/'
    m['path_data_test']    = f'data/input/{data_name}/test/'

    m['path_model']        = f'data/output/{data_name}/{config_name}/'
    m['path_output_pred']  = f'data/output/{data_name}/pred_{config_name}/'
    m['n_sample']          = model_dict['data_pars'].get('n_sample', 5000)

    model_dict[ 'global_pars'] = m
    return model_dict


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name


####################################################################################
config_file    = "cardif_classifier.py"
config_default = 'cardif_lightgbm'


cols_input_type_1 = {
         "coly"   :   "target"
        ,"colid"  :   "ID"
        ,"colcat" :   ["v3","v30", "v31", "v47", "v52", "v56", "v66", "v71", "v74", "v75", "v79", "v91", "v107", "v110", "v112", "v113", "v125"]
        ,"colnum" :   ["v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v23", "v25", "v26", "v27", "v28", "v29", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v48", "v49", "v50", "v51", "v53", "v54", "v55", "v57", "v58", "v59", "v60", "v61", "v62", "v63", "v64", "v65", "v67", "v68", "v69", "v70", "v72", "v73", "v76", "v77", "v78", "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90", "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v108", "v109", "v111", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v126", "v127", "v128", "v129", "v130", "v131"]
        ,"coltext" :  []
        ,"coldate" :  []
        ,"colcross" : ["v3"]
}


cols_input_type_2 = {
         "coly"   :   "target"
        ,"colid"  :   "ID"
        ,"colcat" :   ["v3","v30", "v31", "v47", "v52", ]
        ,"colnum" :   ["v1", "v2", "v4", "v5",    "v108", "v109", "v111", "v114", "v115", "v116", "v117", "v118",  ]
        ,"coltext" :  []
        ,"coldate" :  []
        ,"colcross" : ["v3", "v30"]
}



####################################################################################
##### Params #######################################################################
def cardif_lightgbm(path_model_out="") :
    """
       cardiff
    """
    data_name    = "cardif"
    model_class  = 'LGBMClassifier'
    n_sample     = 5000


    def post_process_fun(y):
        ### After prediction is done
        return  int(y)


    def pre_process_fun(y):
        ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model   #######################################
        ,'model_class': model_class
        ,'model_pars' : {'objective': 'binary',
                               'n_estimators':       100,
                               'learning_rate':      0.01,
                               'boosting_type':      'gbdt',     ### Model hyperparameters
                               'early_stopping_rounds': 5
                        }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun


        ### Before training  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


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

      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': { 'n_sample' : n_sample,
          'cols_input_type' : cols_input_type_2,

          ### family of columns for MODEL  ########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',
          #  'coltext',
          'cols_model_group': [ 'colnum',
                                'colcat_bin',
                                # 'coltext',
                                # 'coldate',
                                # 'colcross_pair'
                              ]

          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
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
def preprocess(config=None, nsample=None):
    model_class  = config  if config is not None else config_default
    mdict        = globals()[model_class]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_preprocess, run_preprocess_old
    run_preprocess.run_preprocess(config_name=  model_class,
                                  path_data         =  m['path_data_train'],
                                  path_output       =  m['path_model'],
                                  path_config_model =  m['config_path'],
                                  n_sample          =  nsample if nsample is not None else m['n_sample'],
                                  mode              =  'run_preprocess')


##################################################################################
########## Train #################################################################
def train(config=None, nsample=None):

    model_class  = config  if config is not None else config_default
    mdict        = globals()[model_class]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_train
    run_train.run_train(config_name=  model_class,
                        path_data_train=  m['path_data_train'],
                        path_output       =  m['path_model'],
                        config_path=  m['config_path'],
                        n_sample          =  nsample if nsample is not None else m['n_sample']
                        )


###################################################################################
######### Check data ##############################################################
def check():
   pass




####################################################################################
####### Inference ##################################################################
def predict(config=None, nsample=None):
    model_class  =  config  if config is not None else config_default
    mdict        = globals()[model_class]()
    m            = mdict['global_pars']


    from source import run_inference,run_inference
    run_inference.run_predict(model_class,
                              path_model  = m['path_model'],
                              path_data   = m['path_data_test'],
                              path_output = m['path_output_pred'],
                              pars={'cols_group': mdict['data_pars']['cols_input_type'],
                                  'pipe_list': mdict['model_pars']['pre_process_pars']['pipe_list']},
                              n_sample    = nsample if nsample is not None else m['n_sample']
                              )


def run_all():
    data_profile()
    preprocess()
    train()
    check()
    predict()




###########################################################################################################
###########################################################################################################
"""
python  cardif_classifier.py  preprocess
python  cardif_classifier.py  train
python  cardif_classifier.py  check
python  cardif_classifier.py  predict
python  cardif_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()
