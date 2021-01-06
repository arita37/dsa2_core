#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### http://localhost:8890/notebooks/da/da/jup_text_features2.ipynb#InitInstall Requirement
get_ipython().system('pip install -r requirements.txt')


# # Init

# In[106]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import gc
import os
import logging
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



warnings.filterwarnings('ignore')


# In[647]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
get_ipython().run_line_magic('matplotlib', 'inline')


import gc, os, sys, copy, string, logging
from datetime import datetime
import warnings
import numpy as np, pandas as pd, sklearn as sk

import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
from tqdm import tqdm_notebook

warnings.filterwarnings('ignore')



# In[648]:


from sklearn import covariance, linear_model, model_selection, preprocessing
from sklearn.cluster import dbscan, k_means
from sklearn.decomposition import PCA, pca, TruncatedSVD, LatentDirichletAllocation, NMF

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, make_scorer,
                             mean_absolute_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[649]:


### Local Import
import util_model
import util_feature
import util_plot
import util_text
import util_date

print(util_feature)




# # Data Loading, basic profiling

# In[698]:


folder = os.getcwd() + "/data/airbnb/"
folder_model = folder + "/models/model01/"
folder_out = folder + "/out"


# In[699]:


df = pd.read_csv(folder+'listings_summary.zip', delimiter=',')

df_list = pd.read_csv(folder+'listings.zip', delimiter=',')
df_rev_sum = pd.read_csv(  folder+'reviews_summary.zip', delimiter=',')





# In[700]:


df.describe()


# In[56]:


df.shape, df.columns, df.dtypes


# In[8]:


### Pandas Profiling for features
# !pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
import pandas_profiling as pp
profile =  df.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file= folder + "pandas_report_output.html")


colexclude = profile.get_rejected_variables(threshold=0.98)
colexclude 


# In[8]:


colexclude = profile.get_rejected_variables(threshold=0.98)
colexclude 


# In[ ]:





# # Column selection by type

# In[20]:


colid = "id"
colnum = [  "review_scores_communication", "review_scores_location", "review_scores_rating"
         ]


colcat = [ "cancellation_policy", "host_response_rate", "host_response_time" ]


coltext = ["house_rules", "neighborhood_overview", "notes", "street"  ]


coldate = [  "calendar_last_scraped", "first_review", "host_since" ]



coly = "price"


colall = colnum + colcat + coltext + coldate

"""

dfnum, dfcat, dfnum_bin, 
dfnum_binhot,  dfcat_hot

colnum, colcat, coltext, 
colnum_bin, colnum_binhot,  

"""

print(colall )


# In[12]:


df = df.set_index( colid )


# In[21]:


df[colall].head(2)


# # Data type normalization, Encoding process (numerics, category)

# In[19]:


#Normalize to NA, NA Handling
# df = df.replace("?", np.nan)

colall


# In[22]:


### colnum procesing 
for x in colnum :
    df[x] = df[x].astype("float32")

print( df[colall].dtypes )


# In[121]:


##### Colcat processing 
colcat_map = pd_colcat_mapping(df, colcat) 
                
#for col in colcat :
#    df[col] =  df[col].apply(lambda x : colcat_map["cat_map"][col].get(x)  )

print( df[colcat].dtypes , colcat_map)


# In[35]:


#coly processing
def pd_str_clean( df, coly) :
    return    df[ coly].apply( lambda x : float( x.replace("$", "")) )

df[ coly ] = pd_str_clean(df, coly)


# # Data Distribution after encoding/ data type normalization

# In[36]:


#### ColTarget Distribution
coly_stat = util_feature.pd_stat_distribution( df[  [ colnum[0] , coly ]] ,  subsample_ratio= 1.0)
coly_stat



# In[ ]:





# In[ ]:





# In[38]:


#### Col numerics distribution
colnum_stat = util_feature.pd_stat_distribution(df[colnum],  subsample_ratio= 0.6)
colnum_stat


# In[ ]:





# In[33]:


#### Col stats distribution
colcat_stat = pd_stat_distribution(df[colcat], subsample_ratio= 0.3)
colcat_stat



# In[ ]:





# In[ ]:





# # Feature processing (strategy 1)

# In[16]:


### BAcKUP data before Pre-processing
dfref = copy.deepcopy( df )
print(dfref.shape)


# In[27]:


"""
Many strategies are possible :
   1) Feature selection before the model
   2) Feature selection using model accuracy.
   
Unless of those cases :
   variable with correl > 99% (ie > imablance class %).
   variable with variance of 0 (as 0).
   Its better to keep all variables at 1st.
   Only Linear model (ie Regression, Logitsic) : 




"""



# ## Colnum

# In[124]:


## Map numerics to Category bin
dfnum, colnum_binmap = pd_colnum_tocat(df, colname=colnum, colexclude=None, colbinmap=None,
                                  bins=5, suffix="_bin",    method="uniform",
                                  return_val="dataframe,param")


print(colnum_binmap)


# In[125]:


colnum_bin =  [  x + "_bin" for x in  list( colnum_map.keys() )   ]
print( colnum_bin )



# In[169]:


dfnum.columns


# In[128]:


### numerics bin to One Hot
dfnum_hot, colnum_onehot = pd_col_to_onehot(dfnum[colnum_bin], colname=colnum_bin,
                             colonehot=None, return_val="dataframe,param")
dfnum_hot[ colnum_onehot   ] .head(10)


# In[202]:


0


# In[277]:


## To pipeline
#  Save each feature processing into "Reproductible pipeline".
#  For Scalability, for repreoduction process
#  Functionnal appraoch --> Can be converted to Spark
#  For scalbility, and paralell

from util_feature import pd_col_to_num, pd_colnum_tocat, pd_col_to_onehot

pipe_preprocess_colnum =[ 
           (util_feature.pd_col_to_num,   {"default": np.nan,} , "Conver string to NA")
           
          ,(util_feature.pd_colnum_tocat, { "colname":None, "colbinmap": colnum_binmap,  'bins': 5, 
                               "method": "uniform", "suffix":"_bin", "return_val": "dataframe"}, 
                               "Convert Numerics to Category " )
           
          ,(util_feature.pd_col_to_onehot, { "colname": None,  "colonehot": colnum_onehot,  
                                "return_val": "dataframe"  } , 
                                "Convert category to onehot" )
]



# In[278]:


###Check pipeline
util_feature.pd_pipeline_apply( df[colnum].iloc[:10000,:], pipe_preprocess_colnum)  




# ## Colcat

# In[236]:


dfcat_hot, colcat_onehot = util_feature.pd_col_to_onehot(df[colcat], colname=colcat,  
                                         colonehot=None, return_val="dataframe,param")
dfcat_hot[colcat_onehot ].head(5)



# In[ ]:





# In[135]:


print("Final features")

dfnum_hot.head(3) 
dfcat_hot.head(3) 


# In[ ]:





# In[289]:


## To pipeline
#  Save each feature processing into "Reproductible pipeline".
#  For Scalability, for repreoduction process
#  Can be easily re-factored for PySpark )
pipe_preprocess_colcat =[ 
           (util_feature.pd_col_fillna, {"value": "-1", "method" : "",   "return_val": "dataframe" },
           )        

          ,(util_feature.pd_col_to_onehot, { "colname": None,  "colonehot": colcat_onehot,  
                                "return_val": "dataframe"  } , "convert to one hot")
]


###Check pipeline
util_feature.pd_pipeline_apply( df[colcat].iloc[:10,:], pipe_preprocess_colcat)  



# In[240]:




os.getcwd()



# In[368]:


string.punctuation


# ## Coltext  

# In[248]:


### Remoe common words
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



# In[148]:


#### NA to ""
df[coltext] =  df[coltext].fillna("")
print( df[coltext].isnull().sum() )


# In[371]:



dftext = util_text.pd_coltext_clean( df[coltext], coltext, stopwords= stopwords ) 
        
dftext.head(6)



# In[391]:


### Word Token List
coltext_freq = {}
for col in coltext :
     coltext_freq[col] =  util_text.pd_coltext_wordfreq(dftext, col) 
    
coltext_freq





# In[374]:


print(coltext_freq["house_rules"].values[:10])


# In[396]:


ntoken=100
dftext_tdidf_dict, word_tokeep_dict = {}, {}
    
for col in coltext:
   word_tokeep = coltext_freq[col]["word"].values[:ntoken]
   word_tokeep = [  t for t in word_tokeep if t not in stopwords   ]
 
   dftext_tdidf_dict[col], word_tokeep_dict[col] = util_text.pd_coltext_tdidf( dftext, coltext= col,  word_minfreq= 1,
                                                        word_tokeep= word_tokeep ,
                                                        return_val= "dataframe,param"  )
  
    
dftext_tdidf_dict, word_tokeep_dict


# In[377]:


0


# In[376]:


0


# In[399]:


dftext_tdidf_dict[col], coltext


# In[403]:


# util_model.



# In[425]:


###  Dimesnion reduction for Sparse Matrix

dftext_svd_list, svd_list  = {},{}
for col in  coltext :
    dftext_svd_list[col], svd_list[col] = util_model.pd_dim_reduction(dftext_tdidf_dict[col], 
                                               colname=None,
                                               model_pretrain=None,                       
                                               colprefix= col + "_svd",
                                               method="svd", 
                                                           dimpca=2, 
                                                           return_val="dataframe,param")



dftext_svd_list, svd_list  
    


# In[426]:



0


# In[321]:





# In[429]:


## To pipeline
#  Save each feature processing into "Reproductible pipeline".
#  For Scalability, for repreoduction process

######### Pipeline ONE
col = 'house_rules'


pipe_preprocess_coltext01 =[ 
           ( util_text.pd_coltext_clean , {"colname": [col], "stopwords"  : stopwords },  )        

          ,( util_text.pd_coltext_tdidf, { "coltext": col,  "word_minfreq" : 1,
                                          "word_tokeep" :  word_tokeep_dict[col],
                                          "return_val": "dataframe"  } , "convert to TD-IDF vector")
    

          ,( util_model.pd_dim_reduction, { "colname": None, 
                                          "model_pretrain" : svd_list[col],
                                          "colprefix": col + "_svd",
                                          "method": "svd", "dimpca" :2, 
                                          "return_val": "dataframe"  } , "Dimension reduction")        
]

### Check pipeline
print( col, word_tokeep )
util_feature.pd_pipeline_apply( df[[col ]].iloc[:10,:], pipe_preprocess_coltext01).iloc[:10, :]  







# In[430]:


######### Pipeline TWO
ntoken= 100
col =  'neighborhood_overview'


pipe_preprocess_coltext02 =[ 
           ( util_text.pd_coltext_clean , {"colname": [col], "stopwords"  : stopwords },  )        

          ,( util_text.pd_coltext_tdidf, { "coltext": col,  "word_minfreq" : 1,
                                          "word_tokeep" :  word_tokeep_dict[col],
                                          "return_val": "dataframe"  } , "convert to TD-IDF vector")
    

          ,( util_model.pd_dim_reduction, { "colname": None, 
                                          "model_pretrain" : svd_list[col],
                                          "colprefix": col + "_svd",
                                          "method": "svd", "dimpca" :2, 
                                          "return_val": "dataframe"  } , "Dimension reduction")        
]

### Check pipeline
print( col, word_tokeep )
util_feature.pd_pipeline_apply( df[[ col ]].iloc[:10, :], pipe_preprocess_coltext02).iloc[:10]  



# In[338]:


dftext[coltext]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Coldate

# In[465]:


coldate = [  "first_review", "host_since" ]


# In[516]:


df[coldate].head(4)



# df[col]  = df[ "first_review" ]


# In[534]:


import dateutil
import copy
from datetime import datetime

pd_datestring_split( df , col, fmt="auto" ).head(5)



# In[519]:


df[coldate].iloc[ :10 , :]


# In[533]:


dfdate_list, coldate_list  = {},{}
for col in  coldate :
    dfdate_list[col] = pd_datestring_split( df , col, fmt="auto", "return_val": "split" )
    coldate_list[col] =  [   t for t in  dfdate_list[col].columns if t not in  [col, col +"_dt"]      ]
    


    dfdate_list, coldate_list


# In[ ]:


######### Pipeline ##########################################
ntoken= 100
col =  'neighborhood_overview'


pipe_preprocess_coltext02 =[ 
           ( util_text.pd_coldate_split , {"colname": col, "fmt": "auto", "return_val": "split"  },  )        
     
]

### Check pipeline
print( col, word_tokeep )
util_feature.pd_pipeline_apply( df[[ col ]].iloc[:10, :], pipe_preprocess_coltext02).iloc[:10]  




# In[552]:



dfdate_hash, coldate_hash_model= util_text.pd_coltext_minhash(df, coldate, n_component=[4, 2], 
                                                    model_pretrain_dict=None,       
                                                    return_val="dataframe,param") 


dfdate_hash, coldate_hash_model


# In[665]:


######### Pipeline ##########################################
pipe_preprocess_coldate_01 =[ 
    (util_text.pd_coltext_fillna , {"colname": coldate, "val" : ""  },  )   ,
    
    (util_text.pd_coltext_minhash , {"colname": coldate, "n_component" : [],
                                          "model_pretrain_dict" : coldate_hash_model,
                                           "return_val": "dataframe"  },  )        
     
]

    
    
### Check pipeline
print( coldate )
util_feature.pd_pipeline_apply( df[ coldate ].iloc[:10, :], pipe_preprocess_coldate_01).iloc[:10]  


# In[523]:







# In[541]:






# In[ ]:





# In[ ]:





# # Pre-Feature Selection

# In[ ]:





# # Train data preparation

# In[ ]:


df[coly] = df[coly].apply( lambda x : 1 if x >90 else -1)


# In[603]:


#### Train data preparation
#dfX = pd.concat(( dfnum_hot, dfcat_hot ), axis=1)

dfX = pd.concat(( # dfnum_hot, 
                  #dfcat_hot, 
                  dfdate_hash,
                  dftext_svd_list['house_rules'],             
                ) , axis=1)

colX = list( dfX.columns )

dfy = df[coly]
#X  = dfX.values
#yy = df[coly].values

Xtrain, Xtest, ytrain, ytest = train_test_split( dfX,  dfy,   
                                                 random_state=42,
                                                 test_size=0.5, shuffle=True)


print( Xtrain.shape, Xtest.shape, ytrain.shape ) 
print( colX )


# In[557]:


dftext_svd_list


# In[572]:





# # Model evaluation

# ## Baseline : Logistic L2 penalty 

# In[670]:



clf_log = sk.linear_model.LogisticRegression(penalty = 'l2' , class_weight = 'balanced')



# In[671]:


clf_log, clf_log_stats = util_model.sk_model_eval_classification(clf_log, 1,
                                           Xtrain, ytrain, Xtest, ytest)


# In[ ]:


clf_log, dd = sk_model_eval_classification(clf_log, 1,
                                           Xtrain, ytrain, Xtest, ytest)


# In[576]:


util_model.sk_model_eval_classification_cv(clf_log,  dfX, dfy, test_size=0.5, ncv=3 )


# In[582]:


clf_log_feat = util_model.sk_feature_impt(clf_log, colX, model_type="logistic" )
clf_log_feat.head(20) 


# In[583]:


clf_log_stats 


# 
# ## Light GBM

# In[584]:



clf_lgb = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l2', 
                         max_depth= 15, n_estimators = 50, objective= 'binary',
                         num_leaves = 38, njobs= -1 )


# In[604]:


clf_lgb, clf_lgb_stats  = util_model.sk_model_eval_classification(clf_lgb, 1,
                                           Xtrain, ytrain, Xtest, ytest)


# In[586]:


import shap
shap.initjs()

#dftest = pd.DataFrame( columns=colX, data=Xtest)

explainer = shap.TreeExplainer( clf_lgb )
shap_values = explainer.shap_values(  Xtest )


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], Xtest.iloc[0,:] )




# In[587]:


# Plot summary_plot as barplot:
shap.summary_plot(shap_values, Xtest)


# In[589]:


# visualize the training set predictions
shap.dependence_plot(  colX[0]  , shap_values, Xtest.iloc[:,:]  )




# In[258]:





# In[593]:


lgb_feature_imp = util_model.sk_feature_impt(clf_lgb.feature_importances_, colname= colX, model_type="lgb")


util_plot.plotbar(lgb_feature_imp.iloc[:10,:], colname=["col", "weight"],  
                  title="feature importance", savefile="lgb_feature_imp.png") 




# In[597]:


kf = StratifiedKFold(n_splits=3, shuffle=True)
# partially based on https://www.kaggle.com/c0conuts/xgb-k-folds-fastai-pca
clf_list = []
for itrain, itest in kf.split(dfX, dfy):
    print("###")
    Xtrain, Xval = dfX.loc[ itrain, : ], dfX.loc[ itest, : ]
    ytrain, yval = dfy.loc[ itrain ], dfy.loc[ itest ]
    clf_lgb.fit(Xtrain, ytrain, eval_set=[(Xval, yval)], 
            early_stopping_rounds=20)
    
    
    clf_list.append( clf_lgb)
    
    



# In[599]:


for i, clfi in enumerate( clf_list) :
    print(i)
    clf_lgbi, dd_lgbi = util_model.sk_model_eval_classification(clfi, 0,
                                               Xtrain, ytrain, Xtest, ytest)

    
clf_lgbi, dd_lgbi


# In[4]:





# In[7]:





# In[ ]:





# In[ ]:





# ## SVM

# In[605]:



clf_svc = SVC(C=1.0, probability=True) # since we need probabilities

clf_svc, clf_svc_stats = util_model.sk_model_eval_classification(clf_svc, 1,
                                               Xtrain, ytrain, Xtest, ytest)


# In[228]:





# In[231]:





# In[ ]:





# In[ ]:





# ## Neural Network MLP Classifier

# In[606]:



from sklearn.neural_network import MLPClassifier

clf_nn = MLPClassifier( hidden_layer_sizes=(50,), max_iter=80, alpha=1e-4,
                        activation="relu",
                        solver='adam', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init= 0.1, early_stopping=True, validation_fraction=0.2 )
                       
                     


# In[608]:


clf_nn, dd_nn = util_model.sk_model_eval_classification(clf_nn, 1,
                                           Xtrain, ytrain, Xtest, ytest)


# In[ ]:





# In[ ]:





# In[ ]:





# # Feature selection
# 

# In[ ]:


### Feature Selection (reduce over-fitting)
   #Pre model feature selection (sometimes some features are useful even with low variance....)
   #Post model feature selection



# In[613]:


### Model independant Selection
colX_kbest= util_model.sk_feature_selection(clf_nn,  method="f_classif", colname= colX, kbest="all",
                                 Xtrain= Xtrain, ytrain= ytrain)


print( colX_kbest )


# In[618]:


colX_best = [ 'first_review_hash_0', 'first_review_hash_1', 
              'first_review_hash_2', 'first_review_hash_3', 
              'host_since_hash_0', 'host_since_hash_1', 'house_rules_svd_0']



# In[614]:


clf_log_feat[ :15 ] 


# In[80]:


clf_log.fit( dfX[colX].values , df[coly].values) 


# In[619]:



feat_eval= util_model.sk_feature_evaluation(clf_log, dfX, 30,  
                                 colname_best=colX_best, dfy= dfy )

feat_eval


# In[620]:


0


# # Ensembling 

# In[ ]:





# In[622]:


from sklearn.ensemble import VotingClassifier

clf_list = []
clf_list.append( ("clf_log", clf_log) )
clf_list.append( ("clf_lgb", clf_lgb) )
clf_list.append( ("clf_svc", clf_svc) )


clf_ens = VotingClassifier(clf_list, voting= "soft")  #Soft is required
print(clf_ens)



# In[624]:


util_model.sk_model_eval_classification(clf_ens, 1,
                                           Xtrain, ytrain, Xtest, ytest)


# In[ ]:





# # Saving models

# In[643]:


var_colname = [   x for x in dir() if  x.startswith("col" )] 


var_df = [   x for x in dir() if  x.startswith("df" )] 

var_pipe = [   x for x in dir() if  x.startswith("pipe" )] 

var_clf = [   x for x in dir() if  x.startswith("clf" )] 


for x in [ "var_colname", "var_df", "var_pipe", "var_clf"   ] :
   print( globals()[x]  , "\n" )




# In[644]:





var_list


# In[675]:


folder_model = folder + "/models/model_01/" 

var_list = var_colname + var_df + var_pipe + var_clf

util.save_all(var_list , folder_model, globals_main= globals() ) 


 


# In[280]:


util.save(clf_log , folder_model + "/clf_predict.pkl") 


# In[651]:


### Validate pkl data
for x in var_list :
    print(x) 
    _ = util.load(  "{a}/{b}.pkl".format(a=folder_model, b=x  ))
   
    
    
        


# # Predict values

# In[659]:


#### Load data
dft = pd.read_csv(folder + 'listings_summary.zip', delimiter=',')

print(dft.shape)
dft.head(3)


# In[182]:





# In[660]:


#### Model folder
print( folder_model )

##### Column names
coly = "price"
colid = "id"


dft = dft.set_index( colid)





# In[ ]:


##### Pre-Process Giobally
dft = dft.replace("?",  np.nan)



# In[194]:


dft.head(3)


# In[195]:


#### Pre-processing  ##############################################
dft_cat = util_feature.pd_pipeline_apply( dft[colcat].iloc[:,:], 
                                            pipe_preprocess_colcat )  



dft_cat.head(4)


# In[196]:


#### Pre-processing   #################################################
dft_num = util_feature.pd_pipeline_apply( dft[colnum].iloc[:,:], 
                                            pipe_preprocess_colnum )  

dft_num.head(4)


# In[667]:


#### Pre-processing   #################################################
coldate = util.load( folder_model + "/coldate.pkl") 
print(coldate)
dft_date = util_feature.pd_pipeline_apply( dft[ coldate ].iloc[:,:], 
                                           pipe_preprocess_coldate_01 )  

dft_date.head(4)


# In[662]:


#### Pre-processing   #################################################
coltext = util.load( folder_model + "/coltext.pkl") 
print(coltext)
dft_text = util_feature.pd_pipeline_apply( dft[coltext].iloc[:,:], 
                                           pipe_preprocess_coltext01 )  

dft_text.head(4)


# In[ ]:





# In[668]:


#### Merge data , Create data
df_final = pd.concat(( #dft_cat, 
                       #dft_num,
                       dft_date,
                       dft_text
                     
                     ), axis=1)

col_final = list( df_final.columns )
df_final.head(5)


colX  = util.load(folder_model + "colX.pkl")
df_final[colX].head(3)



# In[688]:


#### Model Load
clf_prod = util.load(folder_model + "clf_log.pkl")

print(clf_prod)

dfX =  util.load(folder_model + "dfX.pkl")
dfy =  util.load(folder_model + "dfy.pkl")

dfX["ytest_"] =   clf_prod.predict( dfX.iloc[:,:])


print( dfX["ytest_"], dfy )
print(dfy)


# In[689]:


#### Predict  
dft[coly] = clf_prod.predict( df_final[colX].iloc[:,:])
print(coly)






# In[694]:


dft.sample(frac=0.2).head(5)


# In[695]:


dft[coly].sample(frac=0.2).head(10) 


# In[ ]:





# In[696]:


###### Export 
dft = dft.reset_index()
dft[[ colid, coly  ]].to_csv( folder_out + "adress_pred.csv" , index=False, mode="w")



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




