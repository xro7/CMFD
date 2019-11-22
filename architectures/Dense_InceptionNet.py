import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input, Conv2D, Conv2DTranspose, BatchNormalization, Reshape,\
LeakyReLU, Dropout ,ReLU , MaxPool2D, Bidirectional, GRU, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D,Reshape
from tensorflow.keras import Model


class Inception_Layer(Model):
    '''
    Outputs a feature map with the same spatial dimensions and 12 channels
    '''
    def __init__(self,kernels,initial_kernel=1,initial_filter=24,last_filter=6):
        super(Inception_Layer,self).__init__()
        assert isinstance(kernels,tuple)
        self.initial_kernel = initial_kernel
        self.initial_filter = initial_filter
        self.last_filter = last_filter
        self.conv1_1 = Conv2D(self.initial_filter,self.initial_kernel,padding='same')
        self.conv1_2 = Conv2D(self.initial_filter,self.initial_kernel,padding='same')
        self.conv2_1 = Conv2D(self.last_filter,(kernels[0],kernels[0]),padding='same')
        self.conv2_2 = Conv2D(self.last_filter,(kernels[1],kernels[1]),padding='same')
        self.bn1_1 = BatchNormalization()
        self.bn1_2 = BatchNormalization()
        self.bn2_1 = BatchNormalization()
        self.bn2_2 = BatchNormalization()
    
    #@tf.function
    def call(self,x,training=True):
        x1 = self.conv1_1(x)
        x1 = self.bn1_1(x1,training=training)
        x1 = ReLU()(x1)
        x1 = self.conv2_1(x1)
        x1 = self.bn1_2(x1,training=training)
        x1 = ReLU()(x1)
        
        x2 = self.conv1_2(x)
        x2 = self.bn2_1(x2,training=training)
        x2 = ReLU()(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2 (x2,training=training)
        x2 = ReLU()(x2)
        
        concat = Concatenate(axis=-1)([x1,x2])
        return concat

class Pyramid_Extrator_Block(Model):
    '''
    Outputs a feature map with the same spatial dimensions and (12 * num_layers + x.shape[-1]) channels
    '''
    def __init__(self, num_layers, kernels):
        super(Pyramid_Extrator_Block,self).__init__()
        self.num_layers = num_layers
        self.kernels = kernels
        self.inception_layers=[]
        for l in range(self.num_layers):
            self.inception_layers.append(Inception_Layer(kernels))
            
    #@tf.function
    def call(self,x,training=True):
        prev_output = x
        for l in range(self.num_layers):
            output = self.inception_layers[l](prev_output)
            prev_output = Concatenate(axis=-1)([output,prev_output])
        return prev_output
    
class Transition_Block(Model):
    '''
    Pools the spatials dims to with a factor of 2 and filters number of channels
    '''
    def __init__(self, filters, kernel_size=1, pooling_layer=AveragePooling2D,preprocessing_block=False):
        super(Transition_Block,self).__init__()
        self.kernel_size = kernel_size
        self.conv = Conv2D(filters,self.kernel_size,padding='same')
        self.pooling_layer=pooling_layer
        self.preprocessing_block = preprocessing_block
        self.bn = BatchNormalization()
        
    #@tf.function
    def call(self,x,training=True):
        if self.preprocessing_block and x.shape[0:] != (256,256,3):
            x = tf.image.resize(x,(256,256),method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False,antialias=False)
        x = self.conv(x)
        x = self.bn(x,training=training)
        x = ReLU()(x)
        x = self.pooling_layer(pool_size=(2,2),strides=(2,2))(x)
        return x
    
class Dense_InceptionNet(Model):
    
    def __init__(self,return_transition_maps=True,transition_block_filters=[24,36,48,60],
                 pyramid_extractor_layers=[4,5,6],pyramid_extractor_kernels=[(5,7),(3,5),(1,3)],debug=False):
        super(Dense_InceptionNet,self).__init__()
        assert len(transition_block_filters) == len(pyramid_extractor_layers) + 1
        self.return_transition_maps=return_transition_maps
        self.transition_blocks = []
        self.pyramid_extractors = []
        self.transition_block_filters = transition_block_filters
        self.pyramid_extractor_layers = pyramid_extractor_layers
        self.pyramid_extractor_kernels = pyramid_extractor_kernels
        self.debug = debug
        for i in range(len(self.transition_block_filters)):
            if i == 0:
                self.transition_blocks.append(Transition_Block(transition_block_filters[i],kernel_size=9,preprocessing_block=True))  
            else:
                self.transition_blocks.append(Transition_Block(transition_block_filters[i]))
        for i in range(len(self.pyramid_extractor_layers)):
            self.pyramid_extractors.append(Pyramid_Extrator_Block(pyramid_extractor_layers[i],pyramid_extractor_kernels[i]))
    
    def call(self,x,training=True):
        transition_maps = []
        x = self.transition_blocks[0](x,training=training)
        if self.debug: 
            print((x.shape))
        for i in range(len(self.pyramid_extractors)):
            x = self.pyramid_extractors[i](x,training=training)
            x = self.transition_blocks[i+1](x,training=training)
            if self.debug: 
                print((x.shape))
            transition_maps.append(x)
        if self.return_transition_maps:
            return transition_maps
        else:
            return x
        
class Feature_Correlation_Matching(Model):

    def __init__(self, Tl=0.6 ,l=2):
        super(Feature_Correlation_Matching,self).__init__()
        self.Tl = Tl
        self.l = l
        self.epsilon = 1e-9
    
    def call(self,x,training=True):
        predictions = []
        args_predictions = []
        for feature_map in x:
            b,h,w,c = feature_map.shape
            feature_map = tf.reshape(feature_map,(b,-1,c))           
            # http://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
            G = tf.einsum('bik, bjk->bij', feature_map, feature_map)
            D = tf.reshape(tf.linalg.diag_part(G),(b,-1,1))+ tf.transpose(tf.reshape(tf.linalg.diag_part(G),(b,-1,1)),perm=(0,2,1)) - 2*G
            D =  tf.keras.activations.relu(D)  # It is possible  to get negative values in the matrix due to a lack of floating point precision
            D = D+self.epsilon #any zero values in the distance matrix will produce infinite gradients
            norms = tf.sqrt(D)
            prediction = tf.sort(norms,axis=-1)
            arg_prediction = tf.argsort(norms,axis=-1)[:,:,1]
            prediction = tf.where((prediction[:,:,1] / prediction[:,:,2] < self.Tl), 2/(1+tf.exp(prediction[:,:,1])),2/(1+self.l*tf.exp(prediction[:,:,1])))
            prediction = tf.reshape(prediction,(b,h,w,1))
            predictions.append(prediction)
            arg_prediction = tf.reshape(arg_prediction,(b,h,w,1))
            args_predictions.append(arg_prediction)
        return predictions,args_predictions
    
class Hierarchical_Post_Processing(Model):

    def __init__(self):
        super(Hierarchical_Post_Processing,self).__init__()
        self.a = 36 /(36+48+60)
        self.b = 48 /(36+48+60)
        self.h = 60 /(36+48+60)
        #self.conv = Conv2D(1,kernel_size=1,padding='same')
         
    def call(self,x,training=True):
        predictions = []
        for feature_map in x:
            #print(feature_map.shape)
            #feature_map = tf.expand_dims(feature_map,-1)
            prediction = tf.image.resize(feature_map,(256,256),method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False,antialias=False)
            #print(prediction.shape)
            predictions.append(prediction)
        output = self.a *predictions[0] + self.b *predictions[1] + self.h *predictions[2]
        #pred = tf.squeeze(tf.stack(predictions,-1))
        #output = self.conv(pred)
        #output = tf.keras.activations.sigmoid(output)
        return output
    
class CMFD(Model):

    def __init__(self):
        super(CMFD,self).__init__()
        # may i need to preprocess inputs
        self.feature_extractor = Dense_InceptionNet()
        self.correlation_matching = Feature_Correlation_Matching()
        self.post_processing = Hierarchical_Post_Processing()
    
    def call(self,x,training=True):
        x = self.feature_extractor(x,training=training)
        feature_maps,args = self.correlation_matching(x)
        output = self.post_processing(feature_maps)
        return feature_maps,args,output