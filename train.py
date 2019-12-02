import argparse
import os
import itertools
import numpy as np
from training_process import *
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
    parser.add_argument('--save_by_metric', choices=['metric',None], type=str, default=None)
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
#     print(GPU_IDS)
#     physical_gpus = tf.config.experimental.list_physical_devices('GPU')
#     print('physical_gpus: ',physical_gpus)
#     logical_gpus = []
#     for gpu_id in GPU_IDS:
#         logical_gpus.append(physical_gpus[gpu_id])
#     print('logical_gpus: ',logical_gpus)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            #tf.config.experimental.set_visible_devices(gpus[GPU_ID], 'GPU') # use specific cpu
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    
#     if not args.distributed:
#         if logical_gpus:
#             try:
#                 # Currently, memory growth needs to be the same across GPUs
#                 for gpu in logical_gpus:
#                     #allow growth
#                     tf.config.experimental.set_memory_growth(gpu, True)
#                 tf.config.experimental.set_visible_devices(logical_gpus, 'GPU')
#                 logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#                 print(len(physical_gpus), "Physical GPUs,",physical_gpus,'\n', len(logical_gpus), "Logical GPUs",logical_gpus)
#             except RuntimeError as e:
#                 print(e)
    
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
            
        def loss_function(mask,output):

            
            if strategy is  None:
                total_loss = loss(mask,output)
                return total_loss
            else:
                total_loss = tf.reduce_mean(loss(mask,output))

                return total_loss
            
#         def calculate_mask(mask,feature_args):
#             #feature_args = tf.expand_dims(feature_args,-1)

#             b,h,w,c = feature_args.shape
#             resized_mask = tf.image.resize(mask,(h,w),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#             #plt.figure()
#             #plt.imshow(resized_mask.numpy()[0,:,:,0])
#             resized_mask = tf.reshape(resized_mask,(b,-1))
#             feature_args = tf.reshape(feature_args,(b,-1))

#             rearranged_mask = tf.gather(resized_mask, feature_args,axis=1) # use feature args to rearrange resized_mask in axis 1. Note that rearranges each row, batch times res: 8 x8 x64
#             indices = [[i,i] for i in range(b)]
#             rearranged_mask = tf.gather_nd(rearranged_mask, indices) # need to get only the corresponding ararngements. res :8 x 64

#             new_mask = (rearranged_mask + resized_mask) /2.0
#             new_mask = tf.where(resized_mask > 0.0,new_mask,0.0)
#             new_mask = tf.reshape(new_mask,(b,h,w,c))
#             #plt.imshow(new_mask.numpy()[0,:,:,0])
#             return new_mask            
            
        if args.model == 'dense_inception':
            architecture = CMFD()
        else:
            raise Exception('not implemented')
        
        trainer = Trainer(architecture,loss_function,optimizer,train_dataset_iter,train_dataset_size,val_dataset_iter,val_dataset_size,
              args.batch_size,args.epochs,max_patience=100,min_epochs=-1,validation_per_epoch=args.validation_per_epoch,save_by_metric=args.save_by_metric
              ,log_dir='logs/'+args.model+'/'+args.experiment+'_',
              checkpoint_path=os.path.join(args.checkpoint_path,args.experiment,''),restore=args.restore,strategy=strategy)
        trainer.train()
if __name__ == "__main__":
    main()
