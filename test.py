import argparse
import os
import itertools
import numpy as np
from training_process import *
from dataset.dataset import *
from architectures import *
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

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
    
    #python test.py --experiment=test --dataset_root=/home/charitidis/sync/USCISI-CMFD/ --batch_size=8 --gpu_id=0 --learning_rate=1e-3 ---checkpoint_path=/home/charitidis/sync/USCISI-CMFD/ckpts/ --validation_per_epoch=1 --epochs=1 --restore

    args = parser.parse_args()
    print(args)
    

    GPU_IDS = [int(args.gpu_id)]
    print(GPU_IDS)
    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('physical_gpus: ',physical_gpus)
    logical_gpus = []
    for gpu_id in GPU_IDS:
        logical_gpus.append(physical_gpus[gpu_id])
    print('logical_gpus: ',logical_gpus)
    
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
    
    optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
    image_size = [args.image_resize,args.image_resize]
    batch_size = args.batch_size
    dataset_root = args.dataset_root
    test_images = os.path.join(dataset_root,'test','images','')
    test_masks = os.path.join(dataset_root,'test','masks','')
    sort=False
    drop_remainder = False


    test_dataset,test_length = create_dataset(test_images,test_masks,args.batch_size,image_size,extension='.jpg',
                                                    drop_remainder=drop_remainder,sort=sort,shuffle=True)

    test_dataset_iter = iter(test_dataset)
    test_dataset_size = iterator_sizes(test_length,batch_size,drop_remainder=drop_remainder)
    print(test_dataset_size)

          

    loss = tf.keras.losses.BinaryCrossentropy()
            
    def loss_function(mask,maps,args):
        a = 36 /(36+48+60)
        b = 48 /(36+48+60)
        h = 60 /(36+48+60)
        total_loss = 0.0
        losses = []

        for i in range(len(maps)):
            new_mask = calculate_mask(mask,args[i])
            losses.append(loss(new_mask,maps[i]))
            
        total_loss = a*losses[0] + b*losses[1] + h*losses[2] 
        return total_loss

            
    def calculate_mask(mask,feature_args):

        b,h,w,c = feature_args.shape
        resized_mask = tf.image.resize(mask,(h,w),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_mask = tf.reshape(resized_mask,(b,-1))
        feature_args = tf.reshape(feature_args,(b,-1))

        rearranged_mask = tf.gather(resized_mask, feature_args,axis=1) # use feature args to rearrange resized_mask in axis 1. Note that rearranges each row, batch times res: 8 x8 x64
        indices = [[i,i] for i in range(b)]
        rearranged_mask = tf.gather_nd(rearranged_mask, indices) # need to get only the corresponding ararngements. res :8 x 64

        new_mask = (rearranged_mask + resized_mask) /2.0
        new_mask = tf.where(resized_mask > 0.0,new_mask,0.0)
        new_mask = tf.reshape(new_mask,(b,h,w,c))
        return new_mask            
            
    if args.model == 'dense_inception':
        architecture = CMFD()
    else:
        raise Exception('not implemented')
        
    trainer = Trainer(architecture,loss_function,optimizer,test_dataset_iter,test_dataset_size,test_dataset_iter,test_dataset_size,args.batch_size,args.epochs,max_patience=100,min_epochs=-1,
                      validation_per_epoch=args.validation_per_epoch,save_by_metric=args.save_by_metric,log_dir='logs/'+args.model+'/'+args.experiment+'_',
                      checkpoint_path=os.path.join(args.checkpoint_path,args.experiment,''),restore=args.restore,strategy=None)
    timestamp = int(datetime.now().timestamp())
    save_dict = {'step':tf.Variable(0),'epoch':tf.Variable(1,dtype=tf.int16),'val_acc':tf.Variable(-1.0),'val_macro_f1':tf.Variable(-1.0),'timestamp':tf.Variable(timestamp)}   
    ckpt,manager,restore_dict = trainer.create_checkpoint_and_restore(trainer.model,trainer.optimizer,**save_dict)  
    timestamp = restore_dict['timestamp']
    test_log_dir = 'logs/'+args.model+'/'+args.experiment+'_' + str(timestamp) + '/test'
    test_file_writer = tf.summary.create_file_writer(test_log_dir)
    trainer.test(None,test_file_writer)
if __name__ == "__main__":
    main()
