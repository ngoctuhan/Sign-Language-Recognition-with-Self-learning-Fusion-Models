import os, pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

def check_exist_and_make(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

def one_hot(y_train, y_test):
 
    gle = LabelEncoder()
    labels_train = gle.fit_transform(y_train)
    labels_test = gle.transform(y_test)
    mappings = { index: label for index, label in enumerate(gle.classes_)}
    print(mappings)

    label_binary = LabelBinarizer()
    onehot_train = label_binary.fit_transform(labels_train)
    onehot_test = label_binary.transform(labels_test)

    return onehot_train, onehot_test 

def load_process_csv_file(file_csv_train, file_csv_test):

    dataFrame =  pd.read_csv(file_csv_train)
    list_idx = dataFrame['file'].values
    label = dataFrame['label'].values

    dataFrame_test = pd.read_csv(file_csv_test)
    list_idx_test = dataFrame_test['file'].values 
    label_test = dataFrame_test['label'].values 

    onehot_train, onehot_test = one_hot(label, label_test)
    
    return list_idx, onehot_train, list_idx_test, onehot_test

def load_process_csv_file_for_torch(file_csv_train, file_csv_test):

    dataFrame =  pd.read_csv(file_csv_train)
    list_idx = dataFrame['file'].values
    label = dataFrame['label'].values

    dataFrame_test = pd.read_csv(file_csv_test)
    list_idx_test = dataFrame_test['file'].values 
    label_test = dataFrame_test['label'].values 

    gle = LabelEncoder()
    y_train = gle.fit_transform(label)
    y_test = gle.transform(label_test)
    mappings = { index: label for index, label in enumerate(gle.classes_)}
    print(mappings)
    return list_idx, y_train, list_idx_test, y_test