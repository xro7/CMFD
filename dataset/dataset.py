import os
import tensorflow as tf
import numpy as np
import re
import pickle
import pandas as pd
from tqdm import tqdm
import glob

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_image_files_from_dirs(paths,extension='jpg',sort=False):
    
    files=[]
    for path in paths:
        
        manipulation_files = []
        name = path.split('/')[-3]
        if os.path.exists(path):
            for (dirpath, dirnames, filenames) in os.walk(path):
                if sort:
                    filenames.sort(key=natural_keys)
                filenames = [os.path.join(dirpath,i) for i in filenames if extension in i]
                files.extend(filenames)
                manipulation_files.extend(filenames)
            print(path,len(manipulation_files))
        else:
            raise Exception(path+ 'not exists')
    return files

def preprocess_image(image,image_size=[256,256],is_mask=False,crop=True):
    if is_mask:
        image = tf.image.decode_jpeg(image, channels=1)
        #image = tf.cast(image,tf.float32)
        #image = image/255.0
        image = tf.image.convert_image_dtype(image, tf.float32)
#         if crop:
#            image = crop_image_keep_aspect_ratio(image)
        image = tf.image.resize(image, image_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,preserve_aspect_ratio=False)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
        #image = tf.cast(image,tf.float32)
        #image = image/255.0
        image = tf.image.convert_image_dtype(image, tf.float32)
#         if crop:
#             image = crop_image_keep_aspect_ratio(image)
        image = tf.image.resize(image, image_size,method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
  
    return image


def crop_image_keep_aspect_ratio(image):
    h,w,c = image.shape
    min_dim = tf.minimum(h,w)
    return tf.crop_to_bounding_box(0,0,min_dim,min_dim)


def read_image(path,label=None):
    image = tf.io.read_file(path)
    if label is None:
        return image
    else:
        return image,label
    
def load_and_preprocess_image(image_path,map_path,image_size=[256,256]):

    image = read_image(image_path)
    gt_map = read_image(map_path)
    image,mask = preprocess_image(image,image_size=image_size),preprocess_image(gt_map,image_size=image_size,is_mask=True)
    return image,mask

    
# should use map before batch to simplify the process    
def create_tf_dataset(image_paths, map_paths, batch_size=None, image_size=None, repeat=True, shuffle=True,drop_remainder=True):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    maps_ds = tf.data.Dataset.from_tensor_slices(map_paths)
    dataset = tf.data.Dataset.zip((path_ds,maps_ds))
    if shuffle:
        dataset = dataset.shuffle(len(image_paths))
    dataset = dataset.map(lambda x,y: load_and_preprocess_image(x,y,image_size),num_parallel_calls=tf.data.experimental.AUTOTUNE) 
    if batch_size is not None:
        dataset = dataset.batch(batch_size,drop_remainder=drop_remainder)
    
    if repeat:
        dataset = dataset.repeat()
        
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def iterator_sizes(sizes,batch_size,drop_remainder=True):
        
    if isinstance(sizes,tuple) :
        if drop_remainder:
            return list((np.array(sizes)/batch_size).astype(np.int32))
        else:
            remainders =  (np.array(sizes)%batch_size > 0).astype(np.int32)
            return list((np.array(sizes)/batch_size + remainders).astype(np.int32))
    else:
        if drop_remainder:
            return sizes//batch_size
        else:
            if sizes%batch_size > 0:
                return sizes//batch_size +1
            else:
                return sizes//batch_size
            
def create_dataset(image_paths,mask_files,batch_size,image_size,use_cashe=True,cashe_dir='.cashed_data',extension='jpg',drop_remainder=False,sort=False,shuffle=True,files=False):
    
    if not files: 
        print('getting files from specified paths')
        if isinstance(image_paths,str):
            image_paths = [image_paths]

        image_files = get_image_files_from_dirs(image_paths,extension=extension,sort=sort)
        print('image_files',len(image_files))
        #image_files = [file for file in image_files if os.path.exists(file.replace('/images/', '/masks/'))]
        mask_files = [file.replace('/images/', '/masks/')  for file in image_files]
    else:
        image_files = image_paths
    
    print('creating tf.Datasets')
    dataset = create_tf_dataset(image_files,mask_files,batch_size,image_size,repeat=True,shuffle=shuffle,drop_remainder=drop_remainder)
    
    print('Done')
    
    return dataset,len(image_files)

