
import numpy as np
import os 
import tensorflow as tf 
import argparse
from utils.util import check_exist_and_make
from customs.metrics import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model generator from video')

    parser.add_argument('--dataset', type=str, default='VSL',
                            help='Name of dataset')

    parser.add_argument('--saved', type=str, default='checkpoint_cls_gen',
                            help='Saved folder path')
    
    parser.add_argument('--model_path', type=str, default='',
                            help='Path of model classification sensor')

    parser.add_argument('--show', type=bool, default=False,
                            help='Path of model classification sensor')

    parser.add_argument('--batch_size', type=int, default=16,
                            help='Batch size model')

    parser.add_argument('--epoch', type=int, default=100,
                            help='Epoch size model')    
    
    parser.add_argument('--gpu', type=int, default=0,
                            help='GPU use')    
    
    parser.add_argument('--folder', type=str, default='',
                            help='Root of video folder')
    

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    model_encode_video = tf.keras.models.load_model(args.model_path, compile = False)

    from utils.util import load_process_csv_file

    list_idx, onehot_train, list_idx_test, onehot_test = load_process_csv_file(
                'dataset/{}/train.csv'.format(args.dataset), 
                        'dataset/{}/test.csv'.format(args.dataset))

    X_train, X_test = [], []

    for filename in list_idx:

        if args.show:
            print(filename)
        arr_file = np.load(os.path.join(args.folder , filename))
        arr_file = np.expand_dims(arr_file, axis = 0)
        vector_video =  model_encode_video(arr_file)
    
        X_train.append(vector_video[0])

    for filename in list_idx_test:
        if args.show:
            print(filename)
        arr_file = np.load(os.path.join(args.folder, filename))
        arr_file = np.expand_dims(arr_file, axis = 0)
        vector_video =  model_encode_video(arr_file)

        X_test.append(vector_video[0])
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(X_train.shape)
    print(X_test.shape)

    from nets.basenet import mlp_simple
    model = mlp_simple(nb= onehot_train.shape[1])

    model.summary()
    model.compile(
        loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1_m])
    # model.compile(loss='categorical_crossentropy' optimizer='adam' metrics= 'accuracy' )
    filepath=os.path.join(args.saved,"weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5" )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(X_train, onehot_train, batch_size = args.batch_size, epochs = args.epoch, 
            validation_data = (X_test, onehot_test), callbacks=callbacks_list)