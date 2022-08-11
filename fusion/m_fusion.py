import os 
from utils.util import load_process_csv_file
import argparse
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import backend as K
from nets.basenet import mlp_simple

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='Feature fusion between model video generator & model video classification')

        parser.add_argument('--dataset', type=str, default='VSL',
                                help='Name of dataset')

        parser.add_argument('--model_video', type=str, default='',
                                help='Path of model video classification')
        parser.add_argument('--model_gen', type=str, default='',
                                help='Path of generator model')
        parser.add_argument('--half', type=bool, default=False,
                                help='Half sensor using in a model')
        parser.add_argument('--folder', type=str, default=False,
                                help='Folder ')
        parser.add_argument('--gpu', type=str, default="2",
                                help='index GPU')

        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)


        list_idx, onehot_train, list_idx_test, onehot_test = load_process_csv_file(
                'dataset/{}/train.csv'.format(args.dataset), 
                        'dataset/{}/test.csv'.format(args.dataset))

        model_video = tf.keras.models.load_model(args.model_video)
        model_gen = tf.keras.models.load_model(args.model_gen)
      
        
        def predict_last_feature(input_data):
            # with a Sequential model
            get_3rd_layer_output = K.function([model_video.layers[0].input],
                                            [model_video.layers[-3].output])
            layer_output = get_3rd_layer_output([input_data])[0]
            return layer_output
            
        X_train, X_test = [], []
        for filename in list_idx:
                print(filename)
                filename = os.path.join(args.folder, filename)
                arr_frames = np.load(filename)
                arr_frames = np.expand_dims(arr_frames, axis = 0)
                video_output = predict_last_feature(arr_frames)[0]
                feature_gen = model_gen.predict(arr_frames)[0]
                l = np.concatenate((video_output, feature_gen), axis = 0)
                X_train.append(l)

        for filename in list_idx_test:
                filename = os.path.join(args.folder, filename)
                arr_frames = np.load(filename)
                arr_frames = np.expand_dims(arr_frames, axis = 0)
                video_output = predict_last_feature(arr_frames)[0]
                feature_gen = model_gen.predict(arr_frames)[0]
                l = np.concatenate((video_output, feature_gen), axis = 0)
                X_test.append(l)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        print(X_train.shape)
        print(X_test.shape)
       
        model =  mlp_simple(in_shape = (256,),nb= onehot_train.shape[1])

        model.summary()

        model.compile(
        loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
        model.fit(X_train, onehot_train, batch_size = 8, epochs = 150, 
                validation_data = (X_test, onehot_test))

"""
python3 m_fusion.py --gpu 2 --folder .dataset/RGB.npy --model_video Offical_MHAD/no-pretrained/video.hdf5  --model_gen Offical_MHAD/no-pretrained/gen.h5  --dataset UTD-MHAD
"""