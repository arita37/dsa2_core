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
import pandas as pd
import json, copy



####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)



####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)


def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


from util_feature import  save, load_function_uri



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


####################################################################################################
def coltext_stopwords(text, stopwords=None, sep=" "):
    tokens = text.split(sep)
    tokens = [t.strip() for t in tokens if t.strip() not in stopwords]
    return " ".join(tokens)


def pd_coltext_clean( df, col, stopwords= None , pars=None):
    import string, re
    ntoken= pars.get('n_token', 1)
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
    return dftext

def pd_coltext_wordfreq(df, col, stopwords, ntoken=100):
    """
    :param df:
    :param coltext:  text where word frequency should be extracted
    :param nb_to_show:
    :return:
    """
    sep=" "
    coltext_freq=df[col].str.split(expand=True).stack().value_counts()
    coltext_freq=coltext_freq.reset_index()
    print(coltext_freq)
    #coltext_freq = df.apply(lambda x: pd.value_counts(x.split(sep))).sum(axis=0).reset_index()
    coltext_freq.columns = ["word", "freq"]
    print('smo',coltext_freq)
    coltext_freq = coltext_freq.sort_values("freq", ascending=0)
    log(coltext_freq)
                      
    word_tokeep  = coltext_freq["word"].values[:ntoken]
    word_tokeep  = [  t for t in word_tokeep if t not in stopwords   ]

    return coltext_freq, word_tokeep


def nlp_get_stopwords():
    import json
    import string
    stopwords = json.load(open("source/utils/stopwords_en.json") )["word"]
    stopwords = [ t for t in string.punctuation ] + stopwords
    stopwords = [ "", " ", ",", ".", "-", "*", '€', "+", "/" ] + stopwords
    stopwords =list(set( stopwords ))
    stopwords.sort()
    print( stopwords )
    stopwords = set(stopwords)
    return stopwords


def pipe_text(df, col, pars={}):
    from utils import util_text, util_model
    stopwords = pars['stopwords']
    dftext                              = pd_coltext_clean(df, col, stopwords= stopwords , pars=pars)
    print("sht")
    print(dftext)
    coltext_freq, word_tokeep           = pd_coltext_wordfreq(df, col, stopwords, ntoken=100)  ## nb of words to keep
    print(coltext_freq)
    dftext_tdidf_dict, word_tokeep_dict = util_text.pd_coltext_tdidf( dftext, coltext= col,  word_minfreq= 1,
                                                            word_tokeep = word_tokeep ,
                                                            return_val  = "dataframe,param"  )
    log(word_tokeep_dict)
    ###  Dimesnion reduction for Sparse Matrix
    dftext_svd_list, svd_list = util_model.pd_dim_reduction(dftext_tdidf_dict,
                                                   colname        = None,
                                                   model_pretrain = None,
                                                   colprefix      = col + "_svd",
                                                   method         = "svd",  dimpca=2,  return_val="dataframe,param")
    return dftext_svd_list



####################################################################################################
####################################################################################################
def preprocess(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
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

    ##### column names for feature generation #####################################################
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

    #### Pipeline Execution
    pipe_default    = [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
    pipe_list       = preprocess_pars.get('pipe_list', pipe_default)
    pipe_list.append('dfdate')
    pipe_list_pars  = preprocess_pars.get('pipe_pars', [])



    ##### Load data ##############################################################################
    df = load_dataset(path_train_X, path_train_y, colid, n_sample= n_sample)

    ##### Filtering / cleaning rows :   #########################################################
    if "filter" in pipe_list :
        def isfloat(x):
            try :
                a= float(x)
                return 1
            except:
                return 0
        ymin, ymax = filter_pars.get('ymin', -9999999999.0), filter_pars.get('ymax', 999999999.0)
        print(coly)
        df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x))
        print(df['_isfloat'])
        df = df[ df['_isfloat'] > 0 ]
        df = df[df[coly] > ymin]
        df = df[df[coly] < ymax]


    ##### Label processing   ####################################################################
    y_norm_fun = None
    if "label" in pipe_list :
        # Target coly processing, Normalization process  , customize by model
        log("y_norm_fun preprocess_pars")
        y_norm_fun = preprocess_pars.get('y_norm_fun', None)
        if y_norm_fun is not None:
            df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
            save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )
            save_features(df[coly], 'dfy', path_features_store)


    ########### colnum procesing   #############################################################
    for x in colnum:
        print('bam',x)
        df[x] = df[x].astype("float")
    log(df[colall].dtypes)


    if "dfnum" in pipe_list :
        pass


    if "dfnum_norm" in pipe_list :
        log("### colnum normalize  ###############################################################")
        from util_feature import pd_colnum_normalize
        pars = { 'pipe_list': [ {'name': 'fillna', 'naval' : 0.0 }, {'name': 'minmax'} ]}
        dfnum_norm, colnum_norm = pd_colnum_normalize(df, colname=colnum,  pars=pars, suffix = "_norm",
                                                      return_val="dataframe,param")
        log(colnum_norm)
        save_features(dfnum_norm, 'dfnum_norm', path_features_store)


    if "dfnum_bin" in pipe_list :
        log("### colnum Map numerics to Category bin  ###########################################")
        dfnum_bin, colnum_binmap = pd_colnum_tocat(df, colname=colnum, colexclude=None, colbinmap=None,
                                                   bins=10, suffix="_bin", method="uniform",
                                                   return_val="dataframe,param")
        log(colnum_binmap)
        ### Renaming colunm_bin with suffix
        colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
        log(colnum_bin)
        save_features(dfnum_bin, 'dfnum_binmap', path_features_store)


    if "dfnum_hot" in pipe_list and "dfnum_bin" in pipe_list  :
        log("### colnum bin to One Hot")
        dfnum_hot, colnum_onehot = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                                    colonehot=None, return_val="dataframe,param")
        log(colnum_onehot)
        save_features(dfnum_hot, 'dfnum_onehot', path_features_store)


    ##### Colcat processing   ################################################################
    colcat_map = pd_colcat_mapping(df, colcat)
    log(df[colcat].dtypes, colcat_map)

    if "dfcat_hot" in pipe_list :
        log("#### colcat to onehot")
        dfcat_hot, colcat_onehot = pd_col_to_onehot(df[colcat], colname=colcat,
                                                    colonehot=None, return_val="dataframe,param")
        log(dfcat_hot[colcat_onehot].head(5))
        save_features(dfcat_hot, 'dfcat_onehot', path_features_store)



    if "dfcat_bin" in pipe_list :
        log("#### Colcat to integer encoding ")
        dfcat_bin, colcat_bin_map = pd_colcat_toint(df[colcat], colname=colcat,
                                                    colcat_map=None, suffix="_int")
        colcat_bin = list(dfcat_bin.columns)
        save_features(dfcat_bin, 'dfcat_bin', path_features_store)

    if "dfcross_hot" in pipe_list :
        log("#####  Cross Features From OneHot Features   ######################################")
        try :
           df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')
        except :
           df_onehot = copy.deepcopy(dfcat_hot)

        colcross_single_onehot_select = []
        for t in list(df_onehot) :
            for c1 in colcross_single :
                if c1 in t :
                   colcross_single_onehot_select.append(t)

        df_onehot = df_onehot[colcross_single_onehot_select ]
        dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                               pct_threshold=0.02,  m_combination=2)
        log(dfcross_hot.head(2).T)
        colcross_pair_onehot = list(dfcross_hot.columns)
        save_features(dfcross_hot, 'dfcross_onehot', path_features_store)
        del df_onehot ,colcross_pair_onehot

    

    if "dftext" in pipe_list :
        log("##### Coltext processing   ###############################################################")
        stopwords = nlp_get_stopwords()
        pars      = {'n_token' : 100 , 'stopwords': stopwords}
        dftext    = None
        
        for coltext_i in coltext :
            
            ##### Run the text processor on each column text  #############################
            dftext_i = pipe_text( df[[coltext_i ]], coltext_i, pars )
            dftext   = pd.concat((dftext, dftext_i), axis=1)  if dftext is not None else dftext_i
            save_features(dftext_i, 'dftext_' + coltext_i, path_features_store)

        log(dftext.head(6))
        save_features(dftext, 'dftext', path_features_store)



    if "dfdate" in pipe_list :
        log("##### Coldate processing   #############################################################")
        from utils import util_date
        dfdate = None
        for coldate_i in coldate :
            dfdate_i =  util_date.pd_datestring_split( df[[coldate_i]] , coldate_i, fmt="auto", return_val= "split" )
            dfdate  = pd.concat((dfdate, dfdate_i), axis=1)  if dfdate is not None else dfdate_i
            save_features(dfdate_i, 'dfdate_' + coldate_i, path_features_store)
        save_features(dfdate, 'dfdate', path_features_store)
        print('spoo',dfdate)


    ###################################################################################
# ###############
    ##### Save pre-processor meta-parameters
    os.makedirs(path_pipeline_export, exist_ok=True)
    log(path_pipeline_export)
    cols_family = {}

    for t in ['colid',
              "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
              "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
              'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns

              'coldate',
              'coltext',

              "coly", "y_norm_fun"
              ]:
        tfile = f'{path_pipeline_export}/{t}.pkl'
        log(tfile)
        t_val = locals().get(t, None)
        if t_val is not None :
           save(t_val, tfile)
           cols_family[t] = t_val


    ######  Merge AlL  #############################################################################
    dfXy = df[colnum + colcat + [coly] ]
    print('localTT',dfXy)
    for t in [ 'dfnum_bin', 'dfnum_hot', 'dfcat_bin', 'dfcat_hot', 'dfcross_hot',
               'dfdate',  'dftext'  ] :
        if t in locals() :
            print('localT', t, locals()[t])
            dfXy = pd.concat((dfXy, locals()[t] ), axis=1)

    save_features(dfXy, 'dfX', path_features_store)
    colXy = list(dfXy.columns)
    colXy.remove(coly)    ##### Only X columns
    cols_family['colX'] = colXy
    save(colXy, f'{path_pipeline_export}/colsX.pkl' )
    save(cols_family, f'{path_pipeline_export}/cols_family.pkl' )


    ###### Return values  #########################################################################
    return dfXy, cols_family


def preprocess_load(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={}, filter_pars={}, path_features_store=None):
    
    from source.util_feature import load

    dfXy        = pd.read_parquet(path_features_store + "/dfX/features.parquet")

    try :
       dfy  = pd.read_parquet(path_features_store + "/dfy/features.parquet")  
       dfXy = dfXy.join(dfy, on= cols_group['colid']  , how="left") 
    except :
       log('Error no label', path_features_store + "/dfy/features.parquet")
     
    cols_family = load(f'{path_pipeline_export}/cols_family.pkl')

    return  dfXy, cols_family


####################################################################################################
############CLI Command ############################################################################
def run_preprocess(model_name, path_data, path_output, path_config_model="source/config_model.py", n_sample=5000,
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
        dfXy, cols      = preprocess(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_features_store)

    elif mode == "load_preprocess" :
        dfXy, cols      = preprocess_load(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample,
                                 preprocess_pars, filter_pars, path_features_store)


    model_dict['data_pars']['coly'] = cols['coly']
    
    ### Generate actual column names from colum groups : colnum , colcat
    model_dict['data_pars']['cols_model'] = sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , [])                
    log(  model_dict['data_pars']['cols_model'] , model_dict['data_pars']['coly'])
    
   
    log("######### finish #################################", )


if __name__ == "__main__":
    import fire
    fire.Fire()









"""
        pipe_text = load_function_uri( pipe_list_pars['dftext'] )
        pipe_text = None

        if pipe_text is None  :
           stopwords = nlp_get_stopwords()
           pars      = {'n_token' : 100 , 'stopwords': stopwords}
        else :
           stopwords = nlp_get_stopwords()
           pars      = {'n_token' : 100 , 'stopwords': stopwords}
           
           
"""




#################################################################################################
##### Save pre-processor meta-parameters
"""
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
        """

"""
        log("##### Coltext processing   ###############################################################")
        from utils import util_text, util_text_embedding, util_model
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
            df      = df[col].fillna("")
            dftext  = util_text.pd_coltext_clean( df[col], col, stopwords= stopwords )                 
            print(dftext.head(6))
            coltext_freq = util_text.pd_coltext_wordfreq(dftext, col)                                 
            word_tokeep  = coltext_freq[col]["word"].values[:ntoken]
            word_tokeep  = [  t for t in word_tokeep if t not in stopwords   ]
             
            dftext_tdidf_dict, word_tokeep_dict = util_text.pd_coltext_tdidf( dftext, coltext= col,  word_minfreq= 1,
                                                                    word_tokeep = word_tokeep ,
                                                                    return_val  = "dataframe,param"  )
            ###  Dimesnion reduction for Sparse Matrix
            dftext_svd_list, svd_list = util_model.pd_dim_reduction(dftext_tdidf_dict, 
                                                           colname=None,
                                                           model_pretrain=None,                       
                                                           colprefix= col + "_svd",
                                                           method="svd",  dimpca=2,  return_val="dataframe,param")            
            return dftext_svd_list
        pars = {'n_token' : 100 }
        dftext = None
        for coltext_i in coltext :
            dftext_i =   pipe_text( df[[coltext_i ]], coltext_i, pars ) 
            save_features(dftext_i, 'dftext_' + coltext_i, path_features_store)
            dftext  = pd.concat((dftext, dftext_i))  if dftext is not None else dftext_i
"""
