import tensorflow as tf # tf.__version__ == 2.1 
from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import LeakyReLU 
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization


# custom net 
from nets.inception1D import Classifier_INCEPTION
from nets.inception3D import inception3D

def define_sensor_cls_model(input_shape = (300, 12), nb_class=50):
    obj_cls = Classifier_INCEPTION('models', input_shape, nb_class)
    model = obj_cls.build_model(input_shape, nb_class, True)
    return model

def mlp_simple(in_shape=(128,), nb =  27):

    model = Sequential()
    model.add(Dense(1024, input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512) ) 
    model.add(Dense(nb))
    model.add(Activation('softmax'))
    return model 

