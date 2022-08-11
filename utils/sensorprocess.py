import os 
import numpy as np 
import scipy.io

class SensorLoader:

    def __init__(self, steps, save_folder) -> None:
        self.steps = steps 
        self.save_folder = save_folder 

    def normalSizeData(self, data):
        """
        Using zero padding if the data is too short
        """
        if data.shape[0] < self.steps:
            padding_size =(self.steps - data.shape[0], data.shape[1])
            padding = np.zeros(padding_size)
            data_aided = np.concatenate([data, padding], axis = 0)
            return data_aided
        else:
            saved_steps = [int(x * data.shape[0] / self.steps) for x in range(self.steps)]
            data_splited = [data[i] for i in saved_steps]
            return np.array(data_splited)

    def covertMat2Npy(self, folder, filename, saved = False):
        """
        Load file .mat and covert to array 
        """
        file_path =  os.path.join(folder, filename)
        mat = scipy.io.loadmat(file_path)
        data = np.array(mat['d_iner']) # For converting to a NumPy array
        data_normalize = self.normalSizeData(data)
        if saved == True:
            file_saved = os.path.join(self.save_folder, filename.replace('mat', 'npy'))
            np.save(file_saved, data_normalize)
        return data_normalize
