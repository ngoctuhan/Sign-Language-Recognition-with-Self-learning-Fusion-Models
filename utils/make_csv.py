
import pandas as pd 
import os, numpy as np
# from torch import _load_global_deps 

def MHAD_get():
    list_for_train = []
    lb_train = []
    list_for_test = []
    lb_test = []

    root = '.dataset/RGB.npy'
    for filename in os.listdir(root):
        
        components =  filename.split('_')
        action = components[0]
        person =  components[1]

        if person in ['s1', 's3', 's5', 's7']:
            list_for_train.append(filename)
            lb_train.append(action)
        else:
            list_for_test.append(filename)
            lb_test.append(action)

    return list_for_train, lb_train, list_for_test, lb_test

def VSL_get():
    list_for_train = []
    lb_train = []
    list_for_test = []
    lb_test = []

    root = '.dataset/SensorVSL.npy'
    for label in os.listdir(root):
        path_folder_file = os.path.join(root, label)
        try:
            for filename in os.listdir(path_folder_file):
                file_name_full_label = os.path.join(label, filename)
                person = filename.split('_')[0]
                if person in ['N6', 'N7', 'N8', 'N9']:
                    list_for_test.append(file_name_full_label)
                    lb_test.append(label)
                else:
                    list_for_train.append(file_name_full_label)
                    lb_train.append(label)
        except:
            pass 
    return list_for_train, lb_train, list_for_test, lb_test


def Bekery_get():
    
    list_for_train = []
    lb_train = []
    list_for_test = []
    lb_test = []
    folder= ".dataset/RGB"
    for filename in os.listdir(folder):
        name_split = filename.split("_")
        action = name_split[-2]
        person = name_split[-3]

        if person in ['s04', 's06', 's08', 's10']:
            list_for_test.append(filename)
            lb_test.append(action)
        else:
            list_for_train.append(filename)
            lb_train.append(action)

    return list_for_train, lb_train, list_for_test, lb_test

def save_csv(filename, list_file, list_label):

    labels = ['file', 'label']
    list_file= np.array(list_file)
    list_file = np.reshape(list_file, (-1, 1))

    list_label = np.array(list_label)
    list_label = np.reshape(list_label, (-1, 1))
    data_concate = np.concatenate([list_file, list_label], axis=1)

    df = pd.DataFrame(data_concate,columns = labels)
    df.to_csv(filename)

# save_csv('dataset/UTD-MHAD/train.csv', list_for_train, lb_train)
# save_csv('dataset/UTD-MHAD/test.csv', list_for_test, lb_test)

list_for_train, lb_train, list_for_test, lb_test= Bekery_get()
save_csv('dataset/MHAD/train.csv', list_for_train, lb_train)
save_csv('dataset/MHAD/test.csv', list_for_test, lb_test)