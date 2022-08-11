import os 
import numpy as np
import tensorflow as tf

class DataSequence(tf.keras.utils.Sequence):

    '''
    Class use sequence dataset for trainning 
    '''
    def __init__ (self, list_IDs, list_label, folder_video, folder_sensor ,batch_size, 
            shuffle=True, replacement=True, half_sensor = False, dataset = ''):
        
            self.list_IDs = list_IDs          # list name file images
            self.folder_video = folder_video # folder images  
            self.folder_sensor = folder_sensor
            self.batch_size = batch_size      # batch_size
            self.list_label = list_label      # list label of all images
            self.shuffle = shuffle            # shuffle data after epouch true or no
            self.on_epoch_end()               # init shuffle
            self.replace = replacement   # True if using UTH-MHAD
            self.half_sensor = half_sensor
            self.dataset = dataset
        
    def __getitem__(self, index):
        
            if (index + 1) * self.batch_size > len(self.list_IDs):

                batch_index = self.indexes[index*self.batch_size:len(self.list_IDs)] 
            else:
                
                batch_index = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

            batch_file = [self.list_IDs[k] for k in batch_index]
            labels = [self.list_label[k] for k in batch_index]
            videos, sensors = [], []

            for name_file in batch_file:
            # load video
                filename = os.path.join(self.folder_video, name_file)
                arr_frames = np.load(filename)
                videos.append(arr_frames)
                
                if self.replace:

                    if self.dataset == "UTD-MHAD":
                        name_file = name_file.replace('color', 'inertial')
                    else:
                        name_file =  name_file.replace('cam01', 'acc_h01')
                # load sensor
               
                filename = os.path.join(self.folder_sensor, name_file)
                arr_frames = np.load(filename)
                if self.half_sensor == True:
                    chanel =  arr_frames.shape[1]//2
                    sensors.append(arr_frames[:, :chanel])
                else:
                    sensors.append(arr_frames)

            videos = np.array(videos, dtype = np.float32) 
            sensors = np.array(sensors, dtype = np.float32)
            labels = np.array(labels, dtype = np.float32)

            return videos, sensors, labels

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(   len(self.list_IDs) / float(self.batch_size)   ))

    def len(self):
        return len(self.list_IDs)


