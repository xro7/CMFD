import argparse
import os
import itertools
import numpy as np
from training_process import define_metrics, train
from dataset.dataset import *
from architectures import *
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from contextlib import nullcontext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str,help='An experiment name')
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--model', type=str, default='dense_inception',help='architecture')
    parser.add_argument('--restore', default=False, action='store_true')
    parser.add_argument('--save_by_metric', choices=['accuracy','macro-f1'], type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu_id', type=str, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--image_resize', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--validation_per_epoch', type=int, default=10)
    parser.add_argument('--checkpoint_path', type=str, default='ckpts/')
    parser.add_argument('--distributed', default=False, action='store_true')
    
    #python train.py --experiment=test --dataset_root=/data1/USCISI-CMFD/ --batch_size=8 --gpu_id=2 --learning_rate=1e-3 --epochs=200 --validation_per_epoch=1 --checkpoint_path=/home/charitidis/sync/USCISI-CMFD/ckpts/
    
    #python train.py --experiment=test --dataset_root=/data1/USCISI-CMFD/ --batch_size=16 --gpu_id=1,2 --learning_rate=1e-3 --epochs=200 --validation_per_epoch=1 --checkpoint_path=/home/charitidis/sync/USCISI-CMFD/ckpts/ --distributed

    args = parser.parse_args()
    print(args)
    
    if args.distributed:
        ids = args.gpu_id.split(',')
        GPU_IDS = list(map(int, ids))
    else:    
        GPU_IDS = [int(args.gpu_id)]
    print(GPU_IDS)
    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('physical_gpus: ',physical_gpus)
    logical_gpus = []
    for gpu_id in GPU_IDS:
        logical_gpus.append(physical_gpus[gpu_id])
    print('logical_gpus: ',logical_gpus)
    
    if not args.distributed:
        if logical_gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in logical_gpus:
                    #allow growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(logical_gpus, 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(physical_gpus), "Physical GPUs,",physical_gpus,'\n', len(logical_gpus), "Logical GPUs",logical_gpus)
            except RuntimeError as e:
                print(e)
    
    if args.distributed:
        devices=['/gpu:'+str(i) for i in GPU_IDS]
        strategy = tf.distribute.MirroredStrategy(devices)
        scope = strategy.scope()
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    else:
        strategy = None
        scope = nullcontext()        

    with scope:
            
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


        train_dataset,train_length = create_dataset(train_images,train_masks,args.batch_size,image_size,extension='.jpg',
                                                    drop_remainder=drop_remainder,sort=sort,shuffle=True)
        val_dataset,val_length = create_dataset(valid_images,valid_masks,args.batch_size,image_size,extension='.jpg',
                                                drop_remainder=drop_remainder,sort=sort,shuffle=True)
        #test_dataset,length = create_dataset(test_images,test_masks,args.batch_size,image_size,extension='.jpg',drop_remainder=drop_remainder,sort=sort,shuffle=True)

        if strategy is not None:
            train_dataset = strategy.experimental_distribute_dataset(train_dataset)
            val_dataset = strategy.experimental_distribute_dataset(val_dataset)
            #val_dataset = strategy.experimental_distribute_dataset(test_dataset)

        train_dataset_iter = iter(train_dataset)
        train_dataset_size = iterator_sizes(train_length,batch_size,drop_remainder=drop_remainder)
        print(train_dataset_size)
        val_dataset_iter = iter(val_dataset)
        val_dataset_size = iterator_sizes(val_length,batch_size,drop_remainder=drop_remainder)
        print(val_dataset_size)
    #   test_dataset_iter = iter(test_dataset)
    #   test_dataset_size = iterator_sizes(length,batch_size,drop_remainder=drop_remainder)
    #   print(test_dataset_size)
          
        if strategy is not None:
            loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        else:
            loss = tf.keras.losses.BinaryCrossentropy()
            
        def loss_function(gt,maps,from_outputs=False):
            a = 36 /(36+48+60)
            b = 48 /(36+48+60)
            h = 60 /(36+48+60)
            total_loss = 0.0
            losses = []
            if from_outputs:
                total_loss = loss(gt,maps)
            else:
                for i in range(len(maps)):
                    losses.append(loss(gt,maps[i]))
                total_loss = a*losses[0] + b*losses[1] + h*losses[2] 
            if strategy is  None:
                return total_loss
            else:
                #print(total_loss.shape)
                total_loss = tf.reduce_mean(total_loss)
                return total_loss #tf.nn.compute_average_loss(total_loss, global_batch_size=batch_size)
            
        if args.model == 'dense_inception':
            architecture = CMFD()
        else:
            raise Exception('not implemented')

        define_metrics(accuracy=tf.keras.metrics.BinaryAccuracy,loss_aggregation=tf.keras.metrics.Mean)
        train(architecture,loss_function,optimizer,train_dataset_iter,train_dataset_size,val_dataset_iter,val_dataset_size,
              args.batch_size,args.epochs,max_patience=100,min_epochs=-1,validation_per_epoch=args.validation_per_epoch,save_by_metric=args.save_by_metric
              ,log_dir='logs/'+args.model+'/'+args.experiment+'_',
              checkpoint_path=os.path.join(args.checkpoint_path,args.experiment,''),restore=args.restore,strategy=strategy)

if __name__ == "__main__":
    main()
