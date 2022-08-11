import os 
import numpy as np
import tensorflow as tf

class DataSequence(tf.keras.utils.Sequence):

    '''
    Class use sequence dataset for trainning 
    '''
    def __init__ (self, list_IDs, list_lable, folder_video,batch_size, shuffle=True, replacement=True ):
        
        self.list_IDs = list_IDs          # list name file images
        self.folder_video = folder_video # folder images  
        self.batch_size = batch_size      # batch_size
        self.list_label = list_lable      # list label of all images
        self.shuffle = shuffle            # shuffle data after epouch true or no
        self.on_epoch_end()               # init shuffle
        self.replace = replacement   # True if using UTH-MHAD
        
    def __getitem__(self, index):
        
        if (index + 1) * self.batch_size > len(self.list_IDs):

            batch_index = self.indexes[index*self.batch_size:len(self.list_IDs)] 
        else:
            
            batch_index = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        batch_file = [self.list_IDs[k] for k in batch_index]
        
        labels = [self.list_label[k] for k in batch_index]
        videos= []

        for name_file in batch_file:
        # load video
            filename = os.path.join(self.folder_video, name_file)
     
            arr_frames = np.load(filename)
            videos.append(arr_frames)
        
        while len(videos) < self.batch_size:
            videos.append(videos[-1])
            labels.append(labels[-1])

        videos = np.array(videos, dtype = np.float32) 
        # labels = np.array(labels, dtype = np.float32)
        
        return videos, np.array(labels)

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(   len(self.list_IDs) / float(self.batch_size)   ))

    def len(self):
        return len(self.list_IDs)


