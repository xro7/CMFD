import argparse
import os
import itertools
import numpy as np
from training_process import define_metrics, train
from dataset.dataset import *
from architectures import *
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

def loss_function(gt,maps,loss=tf.keras.losses.BinaryCrossentropy()):
    a = 36 /(36+48+60)
    b = 48 /(36+48+60)
    h = 60 /(36+48+60)
    total_loss = 0.0
    losses = []
    for i in range(len(maps)):
        losses.append(loss(gt,maps[i]))
    total_loss = a*losses[0] + b*losses[1] + h*losses[2] 
    return total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str,help='An experiment name')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--model', type=str, default='dense_inception',help='architecture')
    parser.add_argument('--restore', default=False, action='store_true')
    parser.add_argument('--save_by_metric', choices=['accuracy','macro-f1'], type=str, default='accuracy')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--image_resize', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--validation_per_epoch', type=int, default=10)
    parser.add_argument('--checkpoint_path', type=str, default='ckpts/')

    
    #python train.py --experiment=test --dataset_root=/data1/USCISI-CMFD/ --batch_size=8 --gpu_id=2 --learning_rate=1e-3 --epochs=100 --validation_per_epoch=1 --checkpoint_path=/home/charitidis/sync/USCISI-CMFD/ckpts/
    args = parser.parse_args()
    print(args)
    
    GPU_ID = args.gpu_id
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[GPU_ID], 'GPU') # use specific cpu
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
    optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
    image_size = [args.image_resize,args.image_resize]
    batch_size = args.batch_size
    epochs = args.epochs
    
    dataset_root = args.dataset_root
    train_images = os.path.join(dataset_root,'train','images','')
    train_masks = os.path.join(dataset_root,'train','masks','')
    test_images = os.path.join(dataset_root,'test','images','')
    test_masks = os.path.join(dataset_root,'test','masks','')
    valid_images = os.path.join(dataset_root,'valid','images','')
    valid_masks = os.path.join(dataset_root,'valid','masks','')
    sort=False
    drop_remainder = False

    train_dataset,length = create_dataset(train_images,train_masks,args.batch_size,image_size,extension='.jpg',drop_remainder=drop_remainder,sort=sort,shuffle=True)
    train_dataset_iter = iter(train_dataset)
    train_dataset_size = iterator_sizes(length,batch_size,drop_remainder=drop_remainder)
    print(train_dataset_size)

#     test_dataset,length = create_dataset(test_images,test_masks,args.batch_size,image_size,extension='.jpg',drop_remainder=drop_remainder,sort=sort,shuffle=True)
#     test_dataset_iter = iter(test_dataset)
#     test_dataset_size = iterator_sizes(length,batch_size,drop_remainder=drop_remainder)
#     print(test_dataset_size)

    val_dataset,length = create_dataset(valid_images,valid_masks,args.batch_size,image_size,extension='.jpg',drop_remainder=drop_remainder,sort=sort,shuffle=True)
    val_dataset_iter = iter(val_dataset)
    val_dataset_size = iterator_sizes(length,batch_size,drop_remainder=drop_remainder)
    print(val_dataset_size)
    
    if args.model == 'dense_inception':
        architecture = CMFD()
    else:
        raise Exception('not implemented')
    
    define_metrics(accuracy=tf.keras.metrics.BinaryAccuracy,loss_aggregation=tf.keras.metrics.Mean)
    train(architecture,loss_function,optimizer,train_dataset_iter,train_dataset_size,val_dataset_iter,val_dataset_size,
          args.batch_size,args.epochs,max_patience=100,min_epochs=-1,validation_per_epoch=args.validation_per_epoch,save_by_metric=args.save_by_metric
          ,log_dir='logs/'+args.model+'/'+args.experiment+'_',
          checkpoint_path=os.path.join(args.checkpoint_path,args.experiment,''),restore=args.restore)

if __name__ == "__main__":
    main()
