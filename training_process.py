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


class Trainer():

    def __init__(self,model,loss_function,optimizer,train_dataset,train_dataset_size,val_dataset,val_dataset_size,batch_size,epochs,max_patience=100,
          min_epochs=-1,validation_per_epoch=1,save_by_metric='accuracy',log_dir='logs/',checkpoint_path='ckpts/',restore=False,strategy=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.train_dataset_size = train_dataset_size
        self.val_dataset = val_dataset
        self.val_dataset_size = val_dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_patience = max_patience
        self.min_epoches = min_epochs
        self.validation_per_epoch = validation_per_epoch
        self.save_by_metric = save_by_metric
        self.log_dir = log_dir
        self.checkpoint_path = checkpoint_path
        self.restore = restore
        self.strategy = strategy
        self.define_metrics()
        self.epsilon = 1e-12
    

    def define_metrics(self,accuracy=tf.keras.metrics.BinaryAccuracy, loss_aggregation=tf.keras.metrics.Sum, classes=1, test_threshold=0.5):
        
        self.batch_accuracy = accuracy()
        self.train_loss = loss_aggregation()
        self.train_accuracy = accuracy()
        self.test_loss = loss_aggregation()
        self.test_accuracy = accuracy(threshold=test_threshold)
        self.test_PR_AUC = tf.keras.metrics.AUC(curve='ROC')
        self.batch_precision = tf.keras.metrics.Precision()
        self.batch_recall = tf.keras.metrics.Recall()
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.iou = tf.keras.metrics.MeanIoU(num_classes=2)
        
    def f1_score(self, precision, recall):
        f1_score = 2 * (precision * recall) / (precision+recall+ self.epsilon)
        return f1_score
        

    #@tf.function
    def train_step(self,batch,step):

        with tf.GradientTape() as tape:
            images, ground_truth = batch

            feature_maps,feature_args,outputs = self.model(images,training=True)
            batch_loss = self.loss_function(ground_truth,feature_maps,feature_args)
            self.train_loss.update_state(batch_loss)
            self.batch_accuracy.update_state(ground_truth,outputs)
            self.train_accuracy.update_state(ground_truth,outputs)
            self.batch_precision.update_state(ground_truth,outputs)
            self.batch_recall.update_state(ground_truth,outputs)
            self.iou.update_state(ground_truth,outputs)

        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return batch_loss
    
    def log_to_tensorboard(self,batch_loss,step,file_writer=None):
        if file_writer is not None:
            with file_writer.as_default():
                tf.summary.scalar('loss',batch_loss,step=step)
                tf.summary.scalar('accuracy',self.batch_accuracy.result(),step=step)
                tf.summary.scalar('precision',self.batch_precision.result(),step=step)
                tf.summary.scalar('recall',self.batch_recall.result(),step=step)        
                tf.summary.scalar('IoU',self.iou.result(),step=step)     
                tf.summary.scalar('f1',self.f1_score(self.batch_precision.result(),self.batch_recall.result()),step=step)

    #@tf.function
    def test_step(self,batch,step,file_writer=None):
        
        images, ground_truth = batch

        feature_maps,feature_args,outputs  = self.model(images,training=False)
        batch_loss = self.loss_function(ground_truth,feature_maps,feature_args)
        self.test_loss.update_state(batch_loss)
        self.batch_accuracy.update_state(ground_truth,outputs)
        self.test_accuracy.update_state(ground_truth,outputs)
        self.test_PR_AUC.update_state(ground_truth,outputs)
        self.precision.update_state(ground_truth,outputs)
        self.recall.update_state(ground_truth,outputs)
        self.iou.update_state(ground_truth,outputs)
        
        if file_writer is not None:
            with file_writer.as_default():
                 if step % 300 == 0:
                    tf.summary.image("images", images, max_outputs=8, step=step)
                    tf.summary.image("masks", ground_truth, max_outputs=8, step=step)
                    tf.summary.image("outputs", outputs, max_outputs=8, step=step)
        
        return batch_loss

    @tf.function
    def distributed_train_step(self,batch,step):
        per_replica_loss = self.strategy.experimental_run_v2(self.train_step,args=(batch,step))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,axis=None)

    @tf.function
    def distributed_test_step(self,batch,step):
        per_replica_loss = self.strategy.experimental_run_v2(self.test_step, args =(batch,step))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,axis=None)

    def test(self,step=None,file_writer=None):
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        self.test_PR_AUC.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.iou.reset_states()
        bar = tqdm(total=self.val_dataset_size)
        count_steps = False
        if step is None:
            count_steps=True

        try:
            for iteration in range(self.val_dataset_size):
                if count_steps:
                    step = iteration
                batch = next(self.val_dataset)      
                images, ground_truth = batch
                self.batch_accuracy.reset_states()
                if self.strategy is None:
                    batch_loss_result = self.test_step(batch,tf.constant(step,dtype=tf.int64),file_writer)
                else:
                    batch_loss_result = self.distributed_test_step(batch,tf.constant(step,dtype=tf.int64))
                bar.update(1)
                postfix = OrderedDict(loss=f'{batch_loss_result:.4f}', 
                                  accuracy=f'{self.batch_accuracy.result().numpy():.4f}')
                bar.set_postfix(postfix)

        except StopIteration:
            pass

        if file_writer is not None and not count_steps:
            with file_writer.as_default():
                tf.summary.scalar('accuracy',self.test_accuracy.result(),step=step)
                tf.summary.scalar('loss',self.test_loss.result(),step=step)
                tf.summary.scalar('precision',self.precision.result(),step=step)
                tf.summary.scalar('ROC_AUC',self.test_PR_AUC.result(),step=step)
                tf.summary.scalar('recall',self.recall.result(),step=step)
                tf.summary.scalar('f1',self.f1_score(self.precision.result(),self.recall.result()),step=step)
                tf.summary.scalar('IoU',self.iou.result(),step=step)  
        else:
            print('Accuracy: {}'.format(self.test_accuracy.result()))
            print('Loss: {}'.format(self.test_loss.result()))
            print('ROC_AUC: {}'.format(self.test_PR_AUC.result()))
            print('precision: {}'.format(self.precision.result()))
            print('recall: {}'.format(self.recall.result()))
            print('f1',{}.format(self.f1_score(self.precision.result(),self.recall.result())))
            print('IoU: {}'.format(self.iou.result()))
        
    def create_checkpoint_and_restore(self,net,optimizer,choose_ckpt='latest',**kwargs):
        ckpt = tf.train.Checkpoint(net=net,optimizer=optimizer,**kwargs)
        manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)
        save_dict = {}
        if self.restore and manager.latest_checkpoint:
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
                
               
    def train(self):

        timestamp = int(datetime.now().timestamp())
        save_dict = {'step':tf.Variable(0),'epoch':tf.Variable(1,dtype=tf.int16),'val_acc':tf.Variable(-1.0),'val_macro_f1':tf.Variable(-1.0),
                     'timestamp':tf.Variable(timestamp)}   
        ckpt,manager,restore_dict = self.create_checkpoint_and_restore(self.model,self.optimizer,**save_dict)   
        step=restore_dict['step']
        best_val_acc = restore_dict['val_acc']
        init_epoch = restore_dict['epoch']
        timestamp = restore_dict['timestamp']
        best_macro_f1 = restore_dict['val_macro_f1']
        model = restore_dict['model']
        optimizer  = restore_dict['optimizer']
        print(restore_dict)
        #tensorboard
        if self.log_dir is None:
            train_file_writer = None
            val_file_writer = None
            #test_file_writer = None
        else:
            train_log_dir = self.log_dir + str(timestamp) + '/train'
            val_log_dir = self.log_dir + str(timestamp) + '/val'
            #test_log_dir = log_dir + str(timestamp) + '/test'
            train_file_writer = tf.summary.create_file_writer(train_log_dir)
            val_file_writer = tf.summary.create_file_writer(val_log_dir)
            #test_file_writer = tf.summary.create_file_writer(test_log_dir)

        start_training_time = time.time()
        count_patience=0
        stop_training=False
        step_to_validate = self.train_dataset_size//self.validation_per_epoch

        if self.save_by_metric == 'macro-f1':
            metric_to_inspect = test_macro_f1
            current_best_metric = best_val_acc
        elif self.save_by_metric == 'accuracy':
            metric_to_inspect = test_accuracy
            current_best_metric = best_macro_f1
        elif self.save_by_metric is None:
            metric_to_inspect = None
        else:
            raise Exception('Not Implemented metric')

        for epoch in range(init_epoch, self.epochs):
            start_epoch_time = time.time()
            print('EPOCH: {}'.format(epoch)) 
            #reset metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            #train_macro_f1.reset_states()
            bar = tqdm(total=self.train_dataset_size)
            validation_round = 1
            for iteration in range(self.train_dataset_size):
                batch = next(self.train_dataset)
                self.batch_accuracy.reset_states()
                self.batch_precision.reset_states()
                self.batch_recall.reset_states()
                self.iou.reset_states()
                if self.strategy is None:
                    loss = self.train_step(batch,tf.constant(step,dtype=tf.int64))
                else:
                    loss = self.distributed_train_step(batch,tf.constant(step,dtype=tf.int64)) 
                self.log_to_tensorboard(loss,step,train_file_writer)
                step+=1
                bar.update(1)
                postfix = OrderedDict(loss=f'{loss:.4f}', 
                                  accuracy=f'{self.batch_accuracy.result().numpy():.4f}')
                bar.set_postfix(postfix)

                if self.val_dataset is not None:
                    if (iteration+1) % step_to_validate == 0:
                        print('validation round {}/{} for epoch {} at step {}...'.format(validation_round,self.validation_per_epoch,epoch,step))
                        self.test(step,val_file_writer)
                        #print(val_accuracy)
                        validation_round+=1
                
                
            if self.save_by_metric is None:
                ckpt.step.assign(step)
                ckpt.epoch.assign(epoch)
                save_path = manager.save(checkpoint_number=step)
                print("Saved checkpoint for step {}: {}".format(step, save_path))

#             with train_file_writer.as_default():
#                 tf.summary.scalar('loss_per_epoch',self.train_loss.result(),step=epoch)
#                 tf.summary.scalar('accuracy_per_epoch',self.train_accuracy.result(),step=epoch)




            print("EPOCH: ",epoch+1,' finished')
            print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start_epoch_time))
            print ('----------------------------------')
            if stop_training:
                break
        print ('Complete training time is {} sec'.format(time.time()-start_training_time))
        print ('----------------------------------')