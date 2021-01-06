# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
cd analysis
 run preprocess



"""
import warnings
warnings.filterwarnings('ignore')
import sys
import gc
import os
import re
import pandas as pd
import json, copy


####################################################################################################
#### Add path for python import
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(  repo_root + "/")


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)



####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)


from util_feature import  save, load_function_uri



def coltext_stopwords(text, stopwords=None, sep=" "):
    tokens = text.split(sep)
    tokens = [t.strip() for t in tokens if t.strip() not in stopwords]
    return " ".join(tokens)


####################################################################################################
####################################################################################################
from util_feature import  load_dataset



def save_features(df, name, path):
    """

    :param df:
    :param name:
    :param path:
    :return:
    """
    if path is not None :
       os.makedirs( f"{path}/{name}" , exist_ok=True)
       if isinstance(df, pd.Series):
           df0=df.to_frame()
       else:
           df0=df
       df0.to_parquet( f"{path}/{name}/features.parquet")



def text_preprocess(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={}, filter_pars={}, path_features_store=None):
    """

    :param path_train_X:
    :param path_train_y:
    :param path_pipeline_export:
    :param cols_group:
    :param n_sample:
    :param preprocess_pars:
    :param filter_pars:
    :param path_features_store:
    :return:
    """
    from util_feature import (pd_colnum_tocat, pd_col_to_onehot, pd_colcat_mapping, pd_colcat_toint,
                              pd_feature_generate_cross)

    ##### column names for feature generation ###############################################
    log(cols_group)
    coly            = cols_group['coly']  # 'salary'
    colid           = cols_group['colid']  # "jobId"
    colcat          = cols_group['colcat']  # [ 'companyId', 'jobType', 'degree', 'major', 'industry' ]
    colnum          = cols_group['colnum']  # ['yearsExperience', 'milesFromMetropolis']
    
    colcross_single = cols_group.get('colcross', [])   ### List of single columns
    coltext         = cols_group.get('coltext', [])
    coldate         = cols_group.get('coldate', [])
    colall          = colnum + colcat + coltext + coldate
    log(colall)

    ##### Load data ########################################################################
    df = load_dataset(path_train_X, path_train_y, colid, n_sample= n_sample)

    log("##### Coltext processing   ###############################################################")
    from utils import util_text, util_model

    ### Remoe common words  #############################################
    import json
    import string
    punctuations = string.punctuation
    stopwords = json.load(open("stopwords_en.json") )["word"]
    stopwords = [ t for t in string.punctuation ] + stopwords
    stopwords = [ "", " ", ",", ".", "-", "*", '€', "+", "/" ] + stopwords
    stopwords =list(set( stopwords ))
    stopwords.sort()
    print( stopwords )
    stopwords = set(stopwords)

    def pipe_text(df, col, pars={}):
        ntoken= pars['n_token']
        df      = df.fillna("")
        dftext = df
        log(dftext)
        log(col)
        list1 = []
        list1.append(col)
        

        # fromword = [ r"\b({w})\b".format(w=w)  for w in fromword    ]
        # print(fromword)
        for col_n in list1:
            dftext[col_n] = dftext[col_n].fillna("")
            dftext[col_n] = dftext[col_n].str.lower()
            dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.punctuation))
            dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.digits))
            dftext[col_n] = dftext[col_n].apply(lambda x: re.sub("[!@,#$+%*:()'-]", " ", x))

            dftext[col_n] = dftext[col_n].apply(lambda x: coltext_stopwords(x, stopwords=stopwords))              
        
        print(dftext.head(6))
        

        sep=" "
        """
        :param df:
        :param coltext:  text where word frequency should be extracted
        :param nb_to_show:
        :return:
        """
        coltext_freq = df[col].apply(lambda x: pd.value_counts(x.split(sep))).sum(axis=0).reset_index()
        coltext_freq.columns = ["word", "freq"]
        coltext_freq = coltext_freq.sort_values("freq", ascending=0)
        log(coltext_freq)
                          
        word_tokeep  = coltext_freq["word"].values[:ntoken]
        word_tokeep  = [  t for t in word_tokeep if t not in stopwords   ]

        
        dftext_tdidf_dict, word_tokeep_dict = util_text.pd_coltext_tdidf( dftext, coltext= col,  word_minfreq= 1,
                                                                word_tokeep = word_tokeep ,
                                                                return_val  = "dataframe,param"  )
        
        log(word_tokeep_dict)
        ###  Dimesnion reduction for Sparse Matrix
        dftext_svd_list, svd_list = util_model.pd_dim_reduction(dftext_tdidf_dict, 
                                                       colname=None,
                                                       model_pretrain=None,                       
                                                       colprefix= col + "_svd",
                                                       method="svd",  dimpca=2,  return_val="dataframe,param")            
        return dftext_svd_list

    pars = {'n_token' : 100 }
    dftext1 = None
    for coltext_i in coltext :
        dftext_i =   pipe_text( df[[coltext_i ]], coltext_i, pars ) 
        save_features(dftext_i, 'dftext_' + coltext_i, path_features_store)
        dftext1  = pd.concat((dftext1, dftext_i))  if dftext1 is not None else dftext_i
    print(dftext1.head(6))
    dftext1.to_csv(r""+path_features_store+"\dftext.csv", index = False)


    ##################################################################################################
    ##### Save pre-processor meta-parameters
    os.makedirs(path_pipeline_export, exist_ok=True)
    log(path_pipeline_export)
    cols_family = {}

    for t in ['coltext']:
        tfile = f'{path_pipeline_export}/{t}.pkl'
        log(tfile)
        t_val = locals().get(t, None)
        if t_val is not None :
           save(t_val, tfile)
           cols_family[t] = t_val

    return dftext1, cols_family


def run_text_preprocess(model_name, path_data, path_output, path_config_model="source/config_model.py", n_sample=5000,
              mode='run_preprocess',):     #prefix "pre" added, in order to make if loop possible
    """
      Configuration of the model is in config_model.py file

    """
    path_output         = root + path_output
    path_data           = root + path_data
    path_features_store = path_output + "/features_store/"
    path_pipeline_out   = path_output + "/pipeline/"
    path_model_out      = path_output + "/model/"
    path_check_out      = path_output + "/check/"
    path_train_X        = path_data   + "/features*"    ### Can be a list of zip or parquet files
    path_train_y        = path_data   + "/target*"      ### Can be a list of zip or parquet files
    log(path_output)


    log("#### load input column family  ###################################################")
    cols_group = json.load(open(path_data + "/cols_group.json", mode='r'))
    log(cols_group)


    log("#### Model parameters Dynamic loading  ############################################")
    model_dict_fun = load_function_uri(uri_name= path_config_model + "::" + model_name)
    model_dict     = model_dict_fun(path_model_out)   ### params


    log("#### Preprocess  #################################################################")
    preprocess_pars = model_dict['model_pars']['pre_process_pars']
    filter_pars     = model_dict['data_pars']['filter_pars']

    if mode == "run_preprocess" :
        dfXy, cols      = text_preprocess(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_features_store)
    
   
    log("######### finish #################################", )


if __name__ == "__main__":
    import fire
    fire.Fire()

