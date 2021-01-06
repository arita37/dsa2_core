# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas
"""
import copy
import os
from collections import OrderedDict


#############################################################################################
print("os.getcwd", os.getcwd())

def log(*s, n=0, m=1, **kw):
    sspace = "#" * n
    sjump = "\n" * m

    ### Implement Logging
    print(sjump, sspace, s, sspace, flush=True, **kw)

class dict2(object):
    def __init__(self, d):
        self.__dict__ = d



import numpy as np, pandas as pd
from tempfile import gettempdir



############################################################################################################
def tf_dataset(dataset_pars):
    """
        dataset_pars ={ "dataset_id" : "mnist", "batch_size" : 5000, "n_train": 500, "n_test": 500,
                            "out_path" : "dataset/vision/mnist2/" }
        tf_dataset(dataset_pars)


        https://www.tensorflow.org/datasets/api_docs/python/tfds
        import tensorflow_datasets as tfds
        import tensorflow as tf

        # Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.
        print(tfds.list_builders())

        # Construct a tf.data.Dataset
        ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)

        # Build your input pipeline
        ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
        for features in ds_train.take(1):
          image, label = features["image"], features["label"]


        NumPy Usage with tfds.as_numpy
        train_ds = tfds.load("mnist", split="train")
        train_ds = train_ds.shuffle(1024).batch(128).repeat(5).prefetch(10)

        for example in tfds.as_numpy(train_ds):
          numpy_images, numpy_labels = example["image"], example["label"]
        You can also use tfds.as_numpy in conjunction with batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object:

        train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
        numpy_ds = tfds.as_numpy(train_ds)
        numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]


        FeaturesDict({
    'identity_attack': tf.float32,
    'insult': tf.float32,
    'obscene': tf.float32,
    'severe_toxicity': tf.float32,
    'sexual_explicit': tf.float32,
    'text': Text(shape=(), dtype=tf.string),
    'threat': tf.float32,
    'toxicity': tf.float32,
})

    """
    import tensorflow_datasets as tfds

    d          = dataset_pars
    dataset_id = d['dataset_id']
    batch_size = d.get('batch_size', -1)  # -1 neans all the dataset
    n_train    = d.get("n_train", 500)
    n_test     = d.get("n_test", 500)
    out_path   = path_norm(d['out_path'] )
    name       = dataset_id.replace(".","-")
    os.makedirs(out_path, exist_ok=True)


    train_ds =  tfds.as_numpy( tfds.load(dataset_id, split= f"train[0:{n_train}]", batch_size=batch_size) )
    test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )

    # test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )



    print("train", train_ds.shape )
    print("test",  test_ds.shape )


    def get_keys(x):
       if "image" in x.keys() : xkey = "image"
       if "text" in x.keys() : xkey = "text"
       return xkey


    for x in train_ds:
       #print(x)
       xkey =  get_keys(x)
       np.savez_compressed(out_path + f"{name}_train" , X = x[xkey] , y = x.get('label') )


    for x in test_ds:
       #print(x)
       np.savez_compressed(out_path + f"{name}_test", X = x[xkey] , y = x.get('label') )

    print(out_path, os.listdir( out_path ))



def download_googledrive(file_list=[ {  "fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4",  "path_target":  "data/input/download/test.json"}], **kw):
    """
      Use in dataloader with
         "uri": mlmodels.data:donwload_googledrive
         file_list = [ {  "fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4",  "path_target":  "ztest/covid19/test.json"},
                        {  "fileid" :  "GOOGLE URL ID"   , "path_target":  "dataset/test.json"},
                 ]
    """
    import gdown
    import random
    # file_list   = kw.get("file_list")
    target_list = []

    for d in file_list :
      fileid = d["fileid"]
      target = d.get("path_target", "data/input/adonwload/googlefile_" + str(random.randrange(1000) )  )

      os.makedirs(os.path.dirname(target), exist_ok=True)

      url = f'https://drive.google.com/uc?id={fileid}'
      gdown.download(url, target, quiet=False)
      target_list.append( target  )

    return target_list



def download_dtopbox(data_pars):
  """

   dataset/

   Prefix based :
      repo::
      dropbox::

   import_data

   preprocess_data

  download_data({"from_path" :  "tabular",
                        "out_path" :  path_norm("ztest/dataset/text/") } )

  Open URL
     https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAoFh0aO9RqwwROksGgasIha?dl=0


  """
  from cli_code.cli_download import Downloader

  folder = data_pars['from_path']  # dataset/text/

  urlmap = {
     "text" :    "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AADHrhC7rLkd42_CEqK6A9oYa/dataset/text?dl=1&subfolder_nav_tracking=1"
     ,"tabular" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAxZkJTGSumLADzj3B5wbA0a/dataset/tabular?dl=1&subfolder_nav_tracking=1"
     ,"pretrained" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AACL3LHW1USWrvsV5hipw27ia/model_pretrained?dl=1&subfolder_nav_tracking=1"

     ,"vision" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAM4k7rQrkjBo09YudYV-6Ca/dataset/vision?dl=1&subfolder_nav_tracking=1"
     ,"recommender": "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AABIb2JjQ6aQHwfq5CU0ypHOa/dataset/recommender?dl=1&subfolder_nav_tracking=1"

  }

  url = urlmap[folder]

  #prefix = "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/"
  #url= f"{prefix}/AADHrhC7rLkd42_CEqK6A9oYa/{folder}?dl=1&subfolder_nav_tracking=1"

  out_path = data_pars['out_path']

  zipname = folder.split("/")[0]


  os.makedirs(out_path, exist_ok=True)
  downloader = Downloader(url)
  downloader.download(out_path)

  import zipfile
  with zipfile.ZipFile( out_path + "/" + zipname + ".zip" ,"r") as zip_ref:
      zip_ref.extractall(out_path)




####################################################################################


def import_data():
  def sent_generator(TRAIN_DATA_FILE, chunksize):
      import pandas as pd
      reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
      for df in reader:
          val3  = df.iloc[:, 3:4].values.tolist()
          val4  = df.iloc[:, 4:5].values.tolist()
          flat3 = [item for sublist in val3 for item in sublist]
          flat4 = [str(item) for sublist in val4 for item in sublist]
          texts = []
          texts.extend(flat3[:])
          texts.extend(flat4[:])

          sequences  = model.tokenizer.texts_to_sequences(texts)
          data_train = pad_sequences(sequences, maxlen=data_pars["MAX_SEQUENCE_LENGTH"])
          yield [data_train, data_train]

  model.model.fit(sent_generator(data_pars["train_data_path"], batch_size / 2),
                  epochs          = epochs,
                  steps_per_epoch = n_steps,
                  validation_data = (data_pars["data_1_val"], data_pars["data_1_val"]))






def get_dataset(data_pars) :
  """
    path:
    is_local  : Local to the repo
    data_type:
    train : 1/0
    data_source :  ams
  """
  dd = data_pars

  if not d.get('is_local') is None :
      dd['path'] = os_package_root_path(__file__, 0, dd['path'] )


  if dd['train'] :
     df = pd.read_csv(path)



     ### Donwload from external


     ## Get from csv, local


     ## Get from csv, external


     ### Get from external tool


