import tensorflow as tf # tf.__version__ == 2.1 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization

"""
Pre-trained model I3D for RGB image or Opatical Image. 
"""
     
WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}

def conv3d_bn(x,filters,num_frames,num_row, num_col,padding='same', strides=(1, 1, 1),
        use_bias = False, use_activation_fn = True,use_bn = True,name=None, data_format = 1):

    '''
        Conv3d layer and batchnorm and activation
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(filters, kernel_size = (num_frames, num_row, num_col),strides=strides,
        padding=padding,use_bias=use_bias,name=conv_name)(x)
    if use_bn:
        if data_format == 0:
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)
    return x

def inception3D( input_shape = (24,224,224,3), include_top = False, pretrained=  False, nb_class = 50):

    """
    Implement model Inception 3D from paper: Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset 

    Parameters:
        - input_shape: shape of input include: (num_frames, size of image) 
        - include_top: add a top to use for classification with dense into nb_class and softmax layer
        - pretrained: using pre-trained model for classification (Suggest use when classification)
        - nb_class: number class want to recognition 

    Return:
        - Return a model (keras model)
    """

    channel_axis = 4
 
    input_shape = Input(shape = input_shape) # init input with shape 

    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(input_shape, 64, 7, 7, 7, strides=(2, 2, 2), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same')

    x = concatenate([branch_0, branch_1, branch_2, branch_3],axis=channel_axis)

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)
   
    model = Model(input_shape, x)       

    if pretrained:
        from keras.utils.data_utils import get_file
        weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics'] #rgb_imagenet_and_kinetics flow_imagenet_and_kinetics
        model_name = 'rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='model')
        model.load_weights(downloaded_weights_path)
   
    # Covert to define shape or classification 
    x = model.output
    x = AveragePooling3D((3, 7, 7), strides=(1, 1, 1), padding='valid')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    # x = Reshape((32,32,1))(x)

    if include_top == True:
        x = Dense(nb_class)(x)
        x= Activation('softmax')(x)
        
    model = Model(input_shape, x)
    return model