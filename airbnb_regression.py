# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

  python airbnb_regression.py  preprocess
  python airbnb_regression.py  train
  python airbnb_regression.py  predict


"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')


####################################################################################
###### Path ########################################################################
from source import util_feature
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


######################################################################################
## config_file     = "airbnb_regression.py"
config_default  = 'airbnb_lightgbm'



#####################################################################################
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
        m0, width0 = 0.0, 0.01  ## Min, Max
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
cols_input_type_1 = {
	 "coly"     : "price"
	,"colid"    : "id"
	,"colcat"   :  [ "cancellation_policy", "host_response_rate", "host_response_time" ]
	,"colnum"   :  [ "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
	,"coltext"  :  [ "house_rules", "neighborhood_overview", "notes", "street"  ]
	,"coldate"  :  [ "calendar_last_scraped", "first_review", "host_since" ]
	,"colcross" :  [ "review_scores_communication", "review_scores_location", "cancellation_policy", "host_response_rate"]
}


cols_input_type_2 = {
	 "coly"     : "price"
	,"colid"    : "id"
	,"colcat"   : ["host_id", "host_location", "host_response_time","host_response_rate","host_is_superhost","host_neighbourhood","host_verifications","host_has_profile_pic","host_identity_verified","street","neighbourhood","neighbourhood_cleansed", "neighbourhood_group_cleansed","city","zipcode", "smart_location","is_location_exact","property_type","room_type", "accommodates","bathrooms","bedrooms", "beds","bed_type","guests_included","calendar_updated", "license","instant_bookable","cancellation_policy","require_guest_profile_picture","require_guest_phone_verification","scrape_id"]
	,"colnum"   : ["host_listings_count","latitude", "longitude","square_feet","weekly_price","monthly_price", "security_deposit","cleaning_fee","extra_people", "minimum_nights","maximum_nights","availability_30","availability_60","availability_90","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication", "review_scores_location","review_scores_value","calculated_host_listings_count","reviews_per_month"]
	,"coltext"  : ["name","summary", "space","description", "neighborhood_overview","notes","transit", "access","interaction", "house_rules","host_name","host_about","amenities"]
	, "coldate" : ["last_scraped","host_since","first_review","last_review"]
	,"colcross" : ["name","host_is_superhost","is_location_exact","monthly_price","review_scores_value","review_scores_rating","reviews_per_month"]
}


####################################################################################
def airbnb_lightgbm(path_model_out="") :
    """

    """
    data_name    = "airbnb"   ###in data/
    model_name   = 'LGBMRegressor'


    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')


    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')


    #############################################################################
    model_dict = {'model_pars': {'model_class': model_name

        ,'model_path': path_model_out
        ,'model_pars': {'objective': 'huber',

                       }  # lightgbm one
        ,'post_process_fun': post_process_fun
        ,'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

        ### Pipeline for data processing ########################
        'pipe_list': [
            {'uri': 'source/preprocessors.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/preprocessors.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/preprocessors.py::pd_coltext',              'pars': {}, 'cols_family': 'coltext',    'cols_out': 'coltext_svd',    'type': ''             },
            {'uri': 'source/preprocessors.py::pd_coldate',              'pars': {}, 'cols_family': 'coldate',    'cols_out': 'coldate',        'type': ''             },
            {'uri': 'source/preprocessors.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],

        }
    },

    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',   #### sklearm names
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                    },

    'data_pars': {
          'cols_input_type' : cols_input_type_1

         # "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
         # "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
         # 'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
         # 'coldate', #'coltext', 'coltext_svd'

         ,'cols_model_group': [  'colnum'
                                ,'colcat_bin'
                                ,'coltext_svd'
                              ]

         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data

         }}


    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, os_get_function_name() )

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
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_preprocess
    run_preprocess.run_preprocess(config_name       =  config_name,
                                  config_path       =  m['config_path'],
                                  n_sample          =  nsample if nsample is not None else m['n_sample'],

                                  ### Optonal
                                  mode              =  'run_preprocess')


##################################################################################
########## Train #################################################################
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


###################################################################################
######### Check data ##############################################################
def check():
   pass



####################################################################################
####### Inference ##################################################################
def predict(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']


    from source import run_inference
    run_inference.run_predict(config_name = config_name,
                              config_path =  m['config_path'],
                              n_sample    = nsample if nsample is not None else m['n_sample'],

                              #### Optional
                              path_data   = m['path_pred_data'],
                              path_output = m['path_pred_output'],
                              model_dict  = None
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
python  airbnb_regression.py  preprocess
python  airbnb_regression.py  train
python  airbnb_regression.py  check
python  airbnb_regression.py  predict
python  airbnb_regression.py  run_all
"""
if __name__ == "__main__":
        import fire
        fire.Fire()






def airbnb_elasticnetcv(path_model_out=""):
    global model_name
    model_name        = 'ElasticNetCV'

    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'ElasticNetCV'
        , 'model_path': path_model_out
        , 'model_pars': {}  # default ones
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

                        ### Pipeline for data processing.
                       'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot',
                                        'dfdate', 'dftext'

                        ]
                                                     }
                                                         },
    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                                    },
    'data_pars': {
            'cols_input_type' : cols_input_type_1,

            'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]


         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                            }}
    return model_dict



