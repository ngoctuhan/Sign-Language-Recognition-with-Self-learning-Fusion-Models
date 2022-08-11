import numpy as np 
import os
from tqdm import tqdm   
import argparse
from nets.basenet import define_sensor_cls_model
from utils.util import load_process_csv_file
from customs.metrics import * 

def load_data_from_csv(list_idx, args):
    X = []
    with tqdm(total = len(list_idx)) as pbar:
        for filename in list_idx:
            try:
                if args.dataset == 'UTD-MHAD':
                    arr_file = np.load(os.path.join(args.path_of_dataset, filename.replace('color', 'inertial')))
                elif args.dataset == "MHAD":
                    filename = filename.replace('cam01', 'acc_h01')
                    arr_file = np.load(os.path.join(args.path_of_dataset,filename) )
                else:
                    arr_file = np.load(os.path.join(args.path_of_dataset, filename))
                
                X.append(arr_file)
            except Exception as e:
                print(e)
            pbar.update(1)

    return np.array(X)

def training_stage(args):

    list_idx, one_hot_train, list_idx_eval, one_hot_eval = load_process_csv_file(
                'dataset/{}/train.csv'.format(args.dataset), 
                        'dataset/{}/test.csv'.format(args.dataset))
    
    X_train = load_data_from_csv(list_idx, args)
    X_eval  = load_data_from_csv(list_idx_eval, args)
    
    print("[INFO]: Shape of data for training: ", X_train.shape)
    print("[INFO]: Shape of data for eval: ", X_eval.shape)
    
    if args.auto:
        model = define_sensor_cls_model((X_train.shape[1], X_train.shape[2]), nb_class=one_hot_train.shape[1])
    else:
        model = define_sensor_cls_model((args.steps, args.channels), nb_class = one_hot_train.shape[1])
    
    # model.summary()
    model.compile(
    loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', recall_m, precision_m, f1_m])

    # model.compile(loss='categorical_crossentropy' optimizer='adam' metrics= 'accuracy' )
    
    import tensorflow as tf
    if args.auto:
        filepath = "checkpoint/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    else:
        filepath = os.path.join(args.saved, args.dataset+"_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5" )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    if args.auto:
        model.fit(X_train, one_hot_train, batch_size = 8, epochs = 100, 
                    validation_data = (X_eval, one_hot_eval), 
                            callbacks=callbacks_list)
    else:
        model.fit(X_train, one_hot_train, batch_size = args.batchsize, epochs = args.epoch, 
                    validation_data = (X_eval, one_hot_eval), 
                            callbacks=callbacks_list)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training model I1D for sensor data')
   

    parser.add_argument('--dataset', type=str, default='VSL',
                        help='Name o dataset')
    parser.add_argument('--path_of_dataset', type=str, default='dataset/sensor/VSL',
                        help='Path of dataset')
    parser.add_argument('--saved', type=str, default='checkpoint',
                        help='Saved folder path')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epoch', type=int, default=150,
                        help='Number epouch training')
    parser.add_argument('--auto', type=bool, default=True,
                        help='auto fix model with dataset')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU')
                        
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    training_stage(args)
