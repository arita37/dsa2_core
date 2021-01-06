# -*- coding: utf-8 -*-

import copy

##############################################################################################   
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

##############################################################################################  
##############################################################################################  
"""
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       
        ### Al SKLEARN API
        #['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
       mod    = 'models.model_sklearn'


# cols['cols_model'] = cols["colnum_onehot"] + cols["colcat_onehot"] + cols["colcross_onehot"]


"""
        

##############################################################################################  
def elasticnetcv(path_model_out):
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'ElasticNetCV'
        , 'model_path': path_model_out
        , 'model_pars': {}  # default ones
        , 'post_process_fun': post_process_fun 
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,
                               }
                                 },
      'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                       'explained_variance_score', 'r2_score', 'median_absolute_error']
                      },
      'data_pars': {
          'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                  }}
    return model_dict               


def lightgbm(path_model_out) :
    """
      Huber Loss includes L1  regurarlization         
      We test different features combinaison, default params is optimal
    """
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'LGBMRegressor'
        , 'model_path': path_model_out
        , 'model_pars': {'objective': 'huber', }  # default
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                        'explained_variance_score', 'r2_score', 'median_absolute_error']
                      },
    
      'data_pars': {
          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
         
         }}
    return model_dict               



def salary_lightgbm(path_model_out) :
    """
      Huber Loss includes L1  regurarlization         
      We test different features combinaison, default params is optimal
    """
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'LGBMRegressor'
        , 'model_path': path_model_out
        , 'model_pars': {'objective': 'huber', }  # default
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                        'explained_variance_score', 'r2_score', 'median_absolute_error']
                      },
    
      'data_pars': {
          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
         
         }}
    return model_dict    



def titanic_lightgbm(path_model_out) :
    """

       titanic 
    """
    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun(y):
        ### Before the prediction is done      
        return  y.astype('int')

    model_dict = {'model_pars': {'model_class': 'LGBMClassifier'    ## Class name for model_sklearn.py
        , 'model_path': path_model_out
        , 'model_pars': {'objective': 'binary','learning_rate':0.03,'boosting_type':'gbdt' }  # default
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' :  None ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },
    
      'data_pars': {
          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }   ### Filter data
         
         }}
    return model_dict               


def bayesian_pyro_salary(path_model_out) :
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'model_bayesian_pyro'
        , 'model_path': path_model_out
        , 'model_pars': {'input_width': 112, }  # default
        , 'post_process_fun': post_process_fun
                                 
        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,
                               }
                                 },          
                  
      'compute_pars': {'compute_pars': {'n_iter': 1200, 'learning_rate': 0.01}
                     , 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                       'explained_variance_score', 'r2_score', 'median_absolute_error']
                     , 'max_size': 1000000
                     , 'num_samples': 300
       },
      'data_pars': {
          'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]    
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                  }}
    return model_dict               
                
                  
def airbnb_lightgbm(path_model_out) :      #bnb model added here, because run_preprocess is calling it from this file
    """
       airbnb
			   
    """
    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun(y):
        ### Before the prediction is done
        return  y.astype('int')

    model_dict = {'model_pars': {'model_class': 'LGBMRegressor'    ## Class name for model_sklearn.py
        , 'model_path'       : path_model_out
        , 'model_pars'       : {'objective': 'binary','learning_rate':0.1,'boosting_type':'gbdt' }  # default
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  None ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': {
          'cols_input_type' : {
                     "coly"   :   "price"
                    ,"colid"  :   "id"
                    ,"colcat" : [  "host_id", "host_location", "host_response_time","host_response_rate","host_is_superhost","host_neighbourhood","host_verifications","host_has_profile_pic","host_identity_verified","street","neighbourhood","neighbourhood_cleansed", "neighbourhood_group_cleansed","city","zipcode", "smart_location","is_location_exact","property_type","room_type", "accommodates","bathrooms","bedrooms", "beds","bed_type","guests_included","calendar_updated", "license","instant_bookable","cancellation_policy","require_guest_profile_picture","require_guest_phone_verification","scrape_id"]
                    ,"colnum" : [ "host_listings_count","latitude", "longitude","square_feet","weekly_price","monthly_price", "security_deposit","cleaning_fee","extra_people", "minimum_nights","maximum_nights","availability_30","availability_60","availability_90","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication", "review_scores_location","review_scores_value","calculated_host_listings_count","reviews_per_month"]    
                    ,"coltext" : ["name","summary", "space","description", "neighborhood_overview","notes","transit", "access","interaction", "house_rules","host_name","host_about","amenities"]
                    , "coldate" : ["last_scraped","host_since","first_review","last_review"]
                    ,"colcross" : ["name","host_is_superhost","is_location_exact","monthly_price","review_scores_value","review_scores_rating","reviews_per_month"]
	                ,"usdpricescol":["price","weekly_price","monthly_price","security_deposit","cleaning_fee","extra_people"]
     
                   },

          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']
         ,'cols_model':       []  # cols['colcat_model'],
         ,'coly':             []        # cols['coly']
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }   ### Filter data

         }}
    return model_dict        


def glm_salary( path_model_out) :
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')



    model_dict = {'model_pars': {'model_class': 'TweedieRegressor'  # Ridge
        , 'model_path': path_model_out
        , 'model_pars': {'power': 0, 'link': 'identity'}  # default ones
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun }
                                 },
                  'compute_pars': {'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                                   'explained_variance_score',  'r2_score', 'median_absolute_error']
                                  },
      'data_pars': {
          'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]
         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                  }
    }
    return model_dict               
               
                            
def cardif_lightgbm(path_model_out) :      #Cardif model added here, because run_preprocess is calling it from this file
    """
       cardif
    """
    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun(y):
        ### Before the prediction is done
        return  y.astype('int')

    model_dict = {'model_pars': {'model_class': 'LGBMClassifier'    ## Class name for model_sklearn.py
        , 'model_path'       : path_model_out
        , 'model_pars'       : {'objective': 'binary','learning_rate':0.1,'boosting_type':'gbdt' }  # default
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  None ,
                               }
                                 },
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': {
          'cols_input_type' : {
                     "coly"   :   "target"
                    ,"colid"  :   "ID"
                    ,"colcat" :   ["v3","v30", "v31", "v47", "v52", "v56", "v66", "v71", "v74", "v75", "v79", "v91", "v107", "v110", "v112", "v113", "v125"]
                    ,"colnum" :   ["v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v23", "v25", "v26", "v27", "v28", "v29", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v48", "v49", "v50", "v51", "v53", "v54", "v55", "v57", "v58", "v59", "v60", "v61", "v62", "v63", "v64", "v65", "v67", "v68", "v69", "v70", "v72", "v73", "v76", "v77", "v78", "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90", "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v108", "v109", "v111", "v114", "v115", "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v126", "v127", "v128", "v129", "v130", "v131"]
                    ,"coltext" :  []
                    ,"coldate" :  []
                    ,"colcross" : ["v3"]   #when 
                   },

          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']
         ,'cols_model':       []  # cols['colcat_model'],
         ,'coly':             []        # cols['coly']
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }   ### Filter data

         }}
    return model_dict        
                  
#('roc_auc_score', 'accuracy_score','average_precision_score',  'f1_score', 'log_loss)
    

