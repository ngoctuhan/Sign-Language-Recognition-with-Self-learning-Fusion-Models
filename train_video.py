import numpy as np 
import os       
import argparse
from utils.util import load_process_csv_file
from nets.inception3D import inception3D
from customs.sequence_video import DataSequence
from customs.metrics import *
import tensorflow as tf

def training_stage(args):

    list_idx, one_hot_train, list_idx_eval, one_hot_eval = load_process_csv_file(
                'dataset/{}/train.csv'.format(args.dataset), 
                        'dataset/{}/test.csv'.format(args.dataset))

    sequenceTrain = DataSequence(list_IDs=list_idx, list_lable=one_hot_train, 
            folder_video =  args.folder, batch_size=args.batch_size)
    sequenceEval = DataSequence(list_IDs=list_idx_eval, list_lable=one_hot_eval, 
            folder_video =  args.folder, batch_size=args.batch_size)

    model = inception3D(input_shape = (224, 224, 3),include_top = True, pretrained= args.pretrain, nb_class = one_hot_train.shape[1])
   
    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', recall_m, precision_m, f1_m])
   
   
    filepath="checkpoint_video/"+ args.dataset +"/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
   
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    if args.auto:
        model.fit(sequenceTrain, batch_size = 16, epochs = 100, 
                    validation_data = sequenceEval, 
                            callbacks=callbacks_list)
    else:
        model.fit( sequenceTrain, batch_size = args.batchsize, epochs = args.epoch, 
                    validation_data = sequenceEval, 
                            callbacks=callbacks_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trainning model I3D for video_classification')
  
    parser.add_argument('--dataset', type=str, default='VSL',
                        help='Number axis of sensor')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')

    parser.add_argument('--folder', type=str, default='.dataset/SignRGB_VSL.npy',
                        help='Folder of video')

    parser.add_argument('--epoch', type=int, default=150,
                        help='Number epouch training')

    parser.add_argument('--auto', type=bool, default=True,
                        help='auto fix model with dataset')

    parser.add_argument('--pretrain', type=bool, default=False,
                        help='Using pretrain model')

    parser.add_argument('--gpu', type=str, default="0",
                        help='index GPU')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    
    training_stage(args)
