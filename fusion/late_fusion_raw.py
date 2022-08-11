import os 
from utils.util import load_process_csv_file
import argparse
import numpy as np 
import tensorflow as tf 

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='Fusion between model classification & model sensor classification')

        parser.add_argument('--dataset', type=str, default='VSL',
                                help='Name of dataset')

        parser.add_argument('--model_video', type=str, default='',
                                help='Path of model classification video')

        parser.add_argument('--model_sensor', type=str, default='',
                                help='Model pretrain for encode sensor')

        parser.add_argument('--half', type=bool, default=False,
                                help='Half sensor using in a model')

        parser.add_argument('--folder_video', type=str, default="",
                                help='Folder ')

        parser.add_argument('--folder_sensor', type=str, default="",
                                help='Folder ')

        parser.add_argument('--gpu', type=str, default="0",
                                help='index GPU')

        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

        video_path = {'VSL': '.dataset/SignRGB_VSL.npy', 'UTD-MHAD': '.dataset/RGB.npy'}
        sensor_path = {'VSL': '.dataset/SensorVSL.npy', 'UTD-MHAD': '.dataset/Inertial.npy'}

        list_idx, onehot_train, list_idx_test, onehot_test = load_process_csv_file(
                'dataset/{}/train.csv'.format(args.dataset), 
                        'dataset/{}/test.csv'.format(args.dataset))

        video , sensor, fusion = 0, 0, 0

        model_video = tf.keras.models.load_model(args.model_video)
        model_sensor = tf.keras.models.load_model(args.model_sensor)

        for i, filename in enumerate(list_idx_test):
                filename1 = os.path.join(args.folder_video, filename)
                arr_frames = np.load(filename1)
                arr_frames = np.expand_dims(arr_frames, axis = 0)

                video_output = model_video.predict(arr_frames)[0]
                
                if args.dataset == 'UTD-MHAD': 
                        name_file = filename.replace('color', 'inertial')
                        filename = os.path.join(args.folder_sensor, name_file)
                        sensors = np.load(filename)
                        sensors = np.expand_dims(sensors, axis = 0)

                elif args.dataset == 'MHAD': 
                        name_file = filename.replace('cam01', 'acc_h01')
                        filename = os.path.join(args.folder_sensor, name_file)
                        sensors = np.load(filename)
                        sensors = np.expand_dims(sensors, axis = 0)

                else:
                        filename = os.path.join(args.folder_sensor, filename)
                        sensors = np.load(filename)[:, :6]
                        sensors = np.expand_dims(sensors, axis = 0)

                sensor_output = model_sensor.predict(sensors)[0]
                
                if np.argmax(video_output) == np.argmax(onehot_test[i]):
                        video += 1 
                
                if np.argmax(sensor_output) == np.argmax(onehot_test[i]):
                        sensor += 1 

                fusion_vect = (video_output + sensor_output) 
                
                if np.argmax(fusion_vect) == np.argmax(onehot_test[i]):
                        fusion += 1 
                
                print("acc video: ", video)
                print("acc sensor: ", sensor)
                print("acc fusion: ", fusion)
        
        print("acc video: ", video / onehot_test.shape[0])
        print("acc sensor: ", sensor/ onehot_test.shape[0])
        print("acc fusion: ", fusion / onehot_test.shape[0])

"""
python3 late_fusion_raw.py --gpu 1 --folder_video .dataset/RGB.npy --model_video Offical_MHAD/no-pretrained/video.hdf5  --model_sensor Offical_MHAD/no-pretrained/sensor.hdf5  --dataset UTD-MHAD
"""