import  os
import  numpy as np
from    PIL import Image
from    matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import re
import pickle
import time
from tqdm import tqdm
import itertools
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import  tensorflow as tf
from    tensorflow import keras
import tensorflow_addons as tfa

def define_metrics(accuracy=tf.keras.metrics.BinaryAccuracy,loss_aggregation=tf.keras.metrics.Mean,f1_score=tfa.metrics.F1Score,classes=1,test_threshold=0.5):
    global batch_accuracy,train_macro_f1,test_macro_f1,batch_macro_f1
    global train_loss,train_accuracy,test_loss,test_accuracy,test_PR_AUC,precision,recall,batch_precision,batch_recall
    batch_accuracy = accuracy()
    train_loss = loss_aggregation()
    train_accuracy = accuracy()
    test_loss = loss_aggregation()
    test_accuracy = accuracy(threshold=test_threshold)
    batch_macro_f1 = f1_score(num_classes=classes,average='macro')
    train_macro_f1 = f1_score(num_classes=classes,average='macro')
    test_macro_f1 = f1_score(num_classes=classes,average='macro')
    test_PR_AUC = tf.keras.metrics.AUC(curve='ROC')
    batch_precision = tf.keras.metrics.Precision()
    batch_recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

#@tf.function
def train_step(model,loss_function,optimizer,batch,step,*model_args,file_writer=None):
    
    batch_accuracy.reset_states()
    batch_macro_f1.reset_states()
    batch_precision.reset_states()
    batch_recall.reset_states()
    
    with tf.GradientTape() as tape:
        images, ground_truth = batch

        feature_maps,outputs = model(images,training=True,*model_args)
        batch_loss = loss_function(ground_truth,feature_maps)
        batch_accuracy.update_state(ground_truth,outputs)
        train_loss.update_state(batch_loss)
        train_accuracy.update_state(ground_truth,outputs)
        batch_precision.update_state(ground_truth,outputs)
        batch_recall.update_state(ground_truth,outputs)
        
#         actuals = tf.reshape(ground_truth,(-1,1))       
#         preds = tf.reshape(outputs,(-1,1)) 
#         batch_macro_f1.update_state(actuals,preds)
#         train_macro_f1.update_state(actuals,preds)

    gradients = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if file_writer is not None:
        with file_writer.as_default():
            tf.summary.scalar('batch_loss',batch_loss,step=step)
            tf.summary.scalar('batch_accuracy',batch_accuracy.result(),step=step)
            #tf.summary.scalar('batch_macro_f1',batch_macro_f1.result(),step=step)
            tf.summary.scalar('batch_precision',batch_precision.result(),step=step)
            tf.summary.scalar('batch_recall',batch_recall.result(),step=step)
            if step % 500 == 0:
                for i,maps in enumerate(feature_maps):
                    tf.summary.image("maps_"+str(i), maps, max_outputs=16, step=step)
                tf.summary.image("images", images, max_outputs=16, step=step)
                tf.summary.image("masks", ground_truth, max_outputs=16, step=step)
                tf.summary.image("predictions", outputs, max_outputs=16, step=step)
                      
    return batch_loss

#@tf.function
def test_step(model,loss_function,batch,step,*model_args,file_writer=None):
    batch_accuracy.reset_states()
    
    images, ground_truth = batch
    
    feature_maps,outputs = model(images,training=False,*model_args)
    batch_loss = loss_function(ground_truth,feature_maps)
        
    batch_accuracy.update_state(ground_truth,outputs)
    test_loss.update_state(batch_loss)
    test_accuracy.update_state(ground_truth,outputs)
    test_PR_AUC.update_state(ground_truth,outputs)
    precision.update_state(actuals,preds)
    recall.update_state(actuals,preds)
    
#     actuals = tf.reshape(ground_truth,(-1,1))       
#     preds = tf.reshape(outputs,(-1,1)) 
#     test_macro_f1.update_state(actuals,preds)

    
    # in case i need to record test steps evaluation 
    if file_writer is not None:
        with file_writer.as_default():
            tf.summary.scalar('batch_loss',batch_loss,step=step)
            tf.summary.scalar('batch_accuracy',batch_accuracy.result(),step=step) 
    return batch_loss

@tf.function
def distributed_train_step(strategy,model,loss_function,optimizer,batch,step,*model_args,file_writer=None):
    per_replica_loss = strategy.experimental_run_v2(train_step,args=(model,loss_function,optimizer,batch,step,*model_args,),kwargs={'file_writer':file_writer})
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,axis=None)
 
@tf.function
def distributed_test_step(strategy,model,loss_function,batch,step,*model_args,file_writer=None):
    per_replica_loss = strategy.experimental_run_v2(test_step, args =(model,loss_function,batch,step,*model_args,),kwargs={'file_writer':file_writer})
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,axis=None)

#@tf.function
def test(model,loss_function,step,test_dataset,test_dataset_size,batch_size,*model_args,file_writer=None,strategy=None):
    test_loss.reset_states()
    test_accuracy.reset_states()
    #test_macro_f1.reset_states()
    test_PR_AUC.reset_states()
    precision.reset_states()
    recall.reset_states()
        
    bar = tqdm(total=test_dataset_size)
    i=0
    
    try:

        for iteration in range(test_dataset_size):
            batch = next(test_dataset)      
            images, ground_truth = batch
            if strategy is None:
                batch_loss_result = test_step(model,loss_function,batch,step,*model_args,file_writer=None)
            else:
                batch_loss_result = distributed_test_step(strategy,model,loss_function,batch,step,*model_args,file_writer=None)
            bar.update(1)
            postfix = OrderedDict(loss=f'{batch_loss_result:.4f}', 
                              accuracy=f'{batch_accuracy.result():.4f}')
            bar.set_postfix(postfix)
        
    except StopIteration:
        pass
               
    if file_writer is not None:
        with file_writer.as_default():
            tf.summary.scalar('val_accuracy',test_accuracy.result(),step=step)
            tf.summary.scalar('val_loss',test_loss.result(),step=step)
            tf.summary.scalar('val_precision',batch_precision.result(),step=step)
            tf.summary.scalar('val_ROC_AUC',test_PR_AUC.result(),step=step)
            tf.summary.scalar('val_recall',batch_recall.result(),step=step)
            #tf.summary.scalar('val_macro_f1',test_macro_f1.result(),step=step)
    else:
        print('Accuracy: {}'.format(test_accuracy.result()))
        print('Loss: {}'.format(test_loss.result()))
        #print('Macro_f1: {}'.format(test_macro_f1.result()))
        print('PR_AUC: {}'.format(test_PR_AUC.result()))
        print('precision: {}'.format(precision.result()))
        print('recall: {}'.format(recall.result()))
                
     
        
def create_checkpoint_and_restore(restore,net,optimizer,choose_ckpt='latest',ckpt_path='ckpts/',**kwargs):
    ckpt = tf.train.Checkpoint(net=net,optimizer=optimizer,**kwargs)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)
    save_dict = {}
    if restore and manager.latest_checkpoint:
        if choose_ckpt =='latest':
            ckpt.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            ckpt.restore(os.path.join(ckpt_path,'ckpt-'+choose_ckpt))
            print("Restored from {}".format(os.path.join(ckpt_path,'ckpt-'+choose_ckpt)))
        for attr in kwargs:
            save_dict[attr] = getattr(ckpt, attr).numpy()
        save_dict['model'] = net
        save_dict['optimizer'] = optimizer
    else:
        print("Initializing from scratch.")
        for attr in kwargs:
            save_dict[attr] = kwargs[attr].numpy()
        save_dict['model'] = net
        save_dict['optimizer'] = optimizer
    return ckpt,manager,save_dict
                
               
# @tf.function
def train(model,loss_function,optimizer,train_dataset,train_dataset_size,val_dataset,val_dataset_size,batch_size,epochs,*model_args,max_patience=10,
          min_epochs=-1,validation_per_epoch=1,save_by_metric='accuracy',log_dir='logs/',checkpoint_path='ckpts/',restore=False,strategy=None):
    
    timestamp = int(datetime.now().timestamp())
    save_dict = {'step':tf.Variable(0),'epoch':tf.Variable(1,dtype=tf.int16),'val_acc':tf.Variable(-1.0),'val_macro_f1':tf.Variable(-1.0),
                 'timestamp':tf.Variable(timestamp)}   
    ckpt,manager,restore_dict = create_checkpoint_and_restore(restore,net=model,optimizer=optimizer,ckpt_path=checkpoint_path,**save_dict)   
    step=restore_dict['step']
    best_val_acc = restore_dict['val_acc']
    init_epoch = restore_dict['epoch']
    timestamp = restore_dict['timestamp']
    best_macro_f1 = restore_dict['val_macro_f1']
    model = restore_dict['model']
    optimizer  = restore_dict['optimizer']
    
    print(restore_dict)
    
    #tensorboard
    if log_dir is None:
        train_file_writer = None
        val_file_writer = None
        #test_file_writer = None
    else:
        train_log_dir = log_dir + str(timestamp) + '/train'
        val_log_dir = log_dir + str(timestamp) + '/val'
        #test_log_dir = log_dir + str(timestamp) + '/test'
        train_file_writer = tf.summary.create_file_writer(train_log_dir)
        val_file_writer = tf.summary.create_file_writer(val_log_dir)
        #test_file_writer = tf.summary.create_file_writer(test_log_dir)
    
    start_training_time = time.time()
    epsilon = 3e-2
    count_patience=0
    stop_training=False
    step_to_validate = train_dataset_size//validation_per_epoch
#    steps_to_skip = step % train_dataset_size
#   print('steps_to_skip',steps_to_skip)

    if save_by_metric == 'macro-f1':
        metric_to_inspect = test_macro_f1
        current_best_metric = best_val_acc
    elif save_by_metric == 'accuracy':
        metric_to_inspect = test_accuracy
        current_best_metric = best_macro_f1
    elif save_by_metric is None:
        metric_to_inspect = None
    else:
        raise Exception('Not Implemented metric')
        
    for epoch in range(init_epoch, epochs):
        start_epoch_time = time.time()
        print('EPOCH: {}'.format(epoch)) 
        #reset metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        train_macro_f1.reset_states()
        bar = tqdm(total=train_dataset_size)
        validation_round = 1
        for iteration in range(train_dataset_size):
            batch = next(train_dataset)
            if strategy is None:
                loss = train_step(model,loss_function,optimizer,batch,step,*model_args,file_writer=train_file_writer)
            else:
                loss = distributed_train_step(strategy,model,loss_function,optimizer,batch,step,*model_args,file_writer=train_file_writer) 
            step+=1
            bar.update(1)
            postfix = OrderedDict(loss=f'{loss:.4f}', 
                              accuracy=f'{batch_accuracy.result().numpy():.4f}')
            bar.set_postfix(postfix)
            
            if val_dataset is not None:
                if (iteration+1) % step_to_validate == 0:
                    print('validation round {}/{} for epoch {} at step {}...'.format(validation_round,validation_per_epoch,epoch,step))
                    test(model,loss_function,step,val_dataset,val_dataset_size,batch_size,*model_args,
                                                 file_writer=val_file_writer,strategy=strategy)
                    #print(val_accuracy)
                    validation_round+=1
                    if save_by_metric is not None:
                        if current_best_metric<metric_to_inspect.result().numpy():
                            current_best_metric=metric_to_inspect.result().numpy()
                            print('current_best_metric',current_best_metric,'metric_to_inspect',metric_to_inspect.result().numpy())
                            best_val_acc = test_accuracy.result().numpy()
                            best_macro_f1 = test_macro_f1.result().numpy()
                            ckpt.val_acc.assign(best_val_acc)
                            ckpt.val_macro_f1.assign(best_macro_f1)
                            ckpt.step.assign(step)
                            ckpt.epoch.assign(epoch)
                            save_path = manager.save(checkpoint_number=step)
                            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                            count_patience = 0
                        else:
                            count_patience = count_patience + 1
                            print('patience level: {}/{}'.format(count_patience,max_patience))
                            if (count_patience>=max_patience) and (epoch>min_epochs):
                                ckpt.val_acc.assign(test_accuracy.result().numpy())
                                ckpt.val_macro_f1.assign(test_macro_f1.result().numpy())
                                ckpt.step.assign(step)
                                ckpt.epoch.assign(epoch)
                                save_path = manager.save(checkpoint_number=step)
                                print("Saved LAST checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                                print("stop criterion")
                                stop_training=True
                                break
        if save_by_metric is None:
            ckpt.step.assign(step)
            ckpt.epoch.assign(epoch)
            save_path = manager.save(checkpoint_number=step)
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        with train_file_writer.as_default():
            tf.summary.scalar('loss_per_epoch',train_loss.result(),step=epoch)
            tf.summary.scalar('accuracy_per_epoch',train_accuracy.result(),step=epoch)
            tf.summary.scalar('macrof1_per_epoch',train_macro_f1.result(),step=epoch)
            #tf.summary.scalar('lr',optimizer.lr(step),step=epoch)# error
            
            
          
        print("EPOCH: ",epoch+1,' finished')
        print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start_epoch_time))
        print ('----------------------------------')
        if stop_training:
            break
    print ('Complete training time is {} sec'.format(time.time()-start_training_time))
    print ('----------------------------------')