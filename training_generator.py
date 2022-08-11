import os 
from customs.sequence_UTD import DataSequence
from utils.util import load_process_csv_file
from training.pipeline import OurMethod
import argparse

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='Training model generator')

        parser.add_argument('--dataset', type=str, default='VSL',
                                help='The name of dataset')

        parser.add_argument('--saved', type=str, default='checkpoint',
                                help='Saved folder path')

        parser.add_argument('--batchsize', type=int, default = 16,
                                help='Batch size for training')

        parser.add_argument('--epoch', type=int, default=150,
                                help='Number epouch training')

        parser.add_argument('--model_path', type=str, default='',
                                help='Path of model classification sensor')

        parser.add_argument('--model_video_path', type=str, default='',
                                help='Model pretrain for encode video')

        parser.add_argument('--pretrain', type=bool, default=False,
                                help='Model pretrain for encode video')
        
        parser.add_argument('--half', type=bool, default=False,
                                help='Half sensor using in a model')

        parser.add_argument('--gpu', type=str, default="0",
                        help='index GPU')

        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

        video_path = {'VSL': '.dataset/SignRGB_VSL.npy', 'UTD-MHAD': '.dataset/RGB.npy', 'MHAD':'.dataset/RGB'}
        sensor_path = {'VSL': '.dataset/SensorVSL.npy', 'UTD-MHAD': '.dataset/Inertial.npy', "MHAD":'.dataset/Sensor'}

        list_idx, onehot_train, list_idx_test, onehot_test = load_process_csv_file(
                'dataset/{}/train.csv'.format(args.dataset), 
                        'dataset/{}/test.csv'.format(args.dataset))

        data_loader_train = DataSequence(list_IDs=list_idx, list_label=onehot_train, 
                folder_video= video_path[args.dataset], folder_sensor = sensor_path[args.dataset], 
                batch_size=args.batchsize, half_sensor= args.half)

        data_loader_test = DataSequence(list_IDs=list_idx_test, list_label=onehot_test, 
                folder_video=video_path[args.dataset], folder_sensor = sensor_path[args.dataset], 
                batch_size=args.batchsize, half_sensor= args.half)

        om = OurMethod(saved_folder=args.saved, model_sensor=args.model_path, pretrain_video=args.pretrain)
        om.fit(data_loader_train, data_loader_test, args.epoch)