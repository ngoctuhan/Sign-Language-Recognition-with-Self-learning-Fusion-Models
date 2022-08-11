import os 
from utils.util import load_process_csv_file
import argparse
import numpy as np 
import tensorflow as tf 
from tqdm import tqdm

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='Late fusion model generate & model video classification')

        parser.add_argument('--dataset', type=str, default='VSL',
                                help='Name of dataset')

        parser.add_argument('--model_video', type=str, default='',
                                help='Path of model classification video')

        parser.add_argument('--model_gen', type=str, default='',
                                help='Path of model generator')

        parser.add_argument('--model_gen_cls', type=str, default='',
                                help='Path of model classification feature generator')

        parser.add_argument('--half', type=bool, default=False,
                                help='Half sensor using in a model')

        parser.add_argument('--folder', type=str, default=False,
                                help='Folder')
     
        parser.add_argument('--gpu', type=str, default="2",
                                help='index GPU')

        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

        video_path = {'VSL': '.dataset/SignRGB_VSL.npy', 'UTD-MHAD': '.dataset/RGB.npy', 'MHAD':'.dataset/RGB'}
        sensor_path = {'VSL': '.dataset/SensorVSL.npy', 'UTD-MHAD': '.dataset/Inertial.npy', 'MHAD':'.dataset/Sensor_300/Shimmer01'}

        list_idx, onehot_train, list_idx_test, onehot_test = load_process_csv_file(
                'dataset/{}/train.csv'.format(args.dataset), 
                        'dataset/{}/test.csv'.format(args.dataset))

        video, sensor, fusion = 0, 0, 0

        model_video = tf.keras.models.load_model(args.model_video)
        model_gen = tf.keras.models.load_model(args.model_gen)
        model_gen_cls = tf.keras.models.load_model(args.model_gen_cls)
        with tqdm(total = len(list_idx_test)) as pbar:
            for i, filename in enumerate(list_idx_test):
                filename = os.path.join(args.folder, filename)
                arr_frames = np.load(filename)
                arr_frames = np.expand_dims(arr_frames, axis = 0)

                video_output = model_video.predict(arr_frames)[0]
                
                feature_gen = model_gen.predict(arr_frames)
                sensor_output = model_gen_cls.predict(feature_gen)[0]
                
                if np.argmax(video_output) == np.argmax(onehot_test[i]):
                        video += 1 
                
                if np.argmax(sensor_output) == np.argmax(onehot_test[i]):
                        sensor += 1 

                fusion_vect = (video_output + sensor_output) 
                
                if np.argmax(fusion_vect) == np.argmax(onehot_test[i]):
                        fusion += 1 
                pbar.update(1)
        
        print("Acc video: ", video / onehot_test.shape[0])
        print("Acc gen: ", sensor/ onehot_test.shape[0])
        print("Acc fusion: ", fusion / onehot_test.shape[0])

"""
python3 late_fusion.py --gpu 2 --folder .dataset/RGB.npy --model_video Offical_MHAD/pretrained/video.h5  --model_gen Offical_MHAD/no-pretrained/gen.h5  --model_gen_cls Offical_MHAD/no-pretrained/gen_cls.hdf5 --dataset UTD-MHAD
"""