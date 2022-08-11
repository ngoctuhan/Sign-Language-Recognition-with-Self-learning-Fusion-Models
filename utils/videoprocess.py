import cv2, os
import numpy as np 
from utils.util import check_exist_and_make

class Videoto3D:

    def __init__(self, save_folder = '' ,steps = 24, width = 224, height = 224):

        self.width = width
        self.height = height
        self.steps = steps
        self.save_folder = save_folder
        check_exist_and_make(self.save_folder)
      
    def video3D(self, folder_store ,filename, saved = False):
        
        file_path = os.path.join(folder_store, filename)
        cap = cv2.VideoCapture(file_path)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT) # give n frame
        frames = [int(x * nframe / self.steps) for x in range(self.steps)]
        framearray = []
        
        for i in range(self.steps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            _, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (self.width, self.height))
            framearray.append(frame)
          
        if len(framearray) < self.steps and len(framearray) > 15:
            
            dis = self.steps - len(framearray)
            feature = framearray[-1]
            for i in range(dis): 
                framearray.append(feature)
        cap.release()
        framearray = np.asanyarray(framearray)

        if saved == True:
            filesaved_path = os.path.join(self.save_folder, filename.replace('avi', 'npy'))
            np.save(filesaved_path, framearray)
            
        return framearray
