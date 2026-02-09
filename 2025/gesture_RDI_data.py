import os
import numpy as np
import h5py

from sklearn.model_selection import train_test_split

def load_data(data_dir):
    data=[]
    labels=[]
    label_dict={'background': 0,'open': 1, 'close': 2, 'turn': 3, }  # Define the label mapping
    
    for gesture in os.listdir(data_dir):
        print(f"Processing gesture: {gesture}")
        for file in os.listdir(os.path.join(data_dir, gesture)):
            print(f"Processing file: {file}")
            if file.endswith(".h5"):
                h5_file_path = os.path.join(data_dir, gesture, file)
                
                with h5py.File(h5_file_path, 'r') as h5_file:
                    
                    #h5_file['DS1'] shape=(2(RDI/PHD),32,32,100)
                    #print("整理RDI資料")
                    RDI_data = np.array(h5_file['DS1'][0])
                    #print("RDI_data shape before transpose:", RDI_data.shape)  # Output:  32, 32, 100)
                    
                    RDI_data = np.transpose(RDI_data, (2, 0, 1))
                    #print("RDI_data shape after transpose:", RDI_data.shape)  # Output: (100, 32, 32)
                    
                    #print("label:",np.array(h5_file['LABEL'])) 
                    #output:label:[[0] [0] [0] [0] [1] [1]... [0] [0] [0] [0] [0] [0] [0] [0]] shape=(100, 1)
                    raw_labels = np.array(h5_file['LABEL']).flatten()  # Flatten the labels to 1D array
                    #print("raw_labels shape:", raw_labels.shape)  # Output: (100,)
                    #print("raw_labels:", raw_labels)  # Output: [0 0 0 0 1 1 ... 0 0 0 0 0 0 0 0]
                    """
                    for label in raw_labels:
                        print("label:", label,"gesture:", label_dict[gesture])
                    """
                    raw_labels[raw_labels == 1] = label_dict[gesture]  # Map 
                    
                    #print("raw_labels after mapping:", raw_labels)
                    
                    #slide window
                    window_size=20
                    stride_step=5
                    
                    for startidx in range(0, len(RDI_data) - window_size + 1, stride_step):
                        end_idx= startidx + window_size
                        window= RDI_data[startidx:end_idx]
                        window_label = raw_labels[startidx:end_idx]
                        #print("window shape:", len(window),len(window[0]), len(window[0][0]))  # Output: (100, 32, 32)  
                        #print("window_label shape:", len(window_label))  # Output: (100,)
                        
                        if np.all(window_label == 0):
                            true_label = 0
                        else:
                            """

                            #appear most frequently one will be label
                            #1.know it is not 0
                            #2.pick up non 0 labels 
                            #3.get the most possible label in the window
                            not_0_label = window_label[window_label != 0]
                            true_label = np.bincount(not_0_label).argmax()  # Get the most common label in the window
                            #np.bincount(not_0_label) counts the occurrences of each label
                            """
                            
                            #middle one will be label
                            true_label = window_label[len(window_label) // 2]  # Get the middle label in the window
                            
                        data.append(window)
                        labels.append(true_label)
    
    data = np.array(data)
    labels = np.array(labels)
                            
    print("\ndata shape", data.shape, "labels shape", labels.shape)

    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels
    
def split_data(data, labels, test_size=0.3, validation_size=0.3):
    
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=validation_size, random_state=42) 
    
    print("\nData split successfully!")
    
    return data_train, labels_train, data_val, labels_val, data_test, labels_test

def save_data(train_data, train_labels, val_data, val_labels, test_data, test_labels, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, 'train.npz'), data=train_data, labels=train_labels)
    np.savez_compressed(os.path.join(output_dir, 'val.npz'), data=val_data, labels=val_labels)
    np.savez_compressed(os.path.join(output_dir, 'test.npz'), data=test_data, labels=test_labels)
    
    print("\nData saved successfully!")

current_path = os.path.dirname(os.path.abspath(__file__))
load_data_path = os.path.join(current_path, 'recorder')
data, labels = load_data(load_data_path)


data_train_set, labels_train_set, data_val_set, labels_val_set, data_test_set, labels_test_set = split_data(data, labels, )#test_size=0.3, validation_size=0.3

save_data(data_train_set, labels_train_set, data_val_set, labels_val_set, data_test_set, labels_test_set, os.path.join(current_path, 'processed_data'))

