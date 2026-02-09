import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers

current_path = os.path.dirname(os.path.abspath(__file__))

dense_hidden_units = 32
dropout_rate = 0.2
#epochs = 
LSTM_units = 64


def model_struct(LSTM_units, dense_hidden_units, dropout_rate, num_classes, time_steps, width, height):
    
    input_shape = (time_steps, width, height, 1)
    sequential_model = tf.keras.Sequential([
        #timedistributed is to remain the time structure of the input data
        layers.TimeDistributed(layers.Conv2D(32, (7, 7), padding='same', activation='relu'),input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((3, 3))),
        #layers.Dropout(0.3),
    
        layers.TimeDistributed(layers.Conv2D(32, (5, 5), padding='same', activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        #layers.Dropout(0.3),
    
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        #layers.Dropout(0.3),
    
        layers.TimeDistributed(layers.Flatten()),
        
        layers.LSTM(LSTM_units),
        
        layers.Dense(dense_hidden_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    return sequential_model

def evaluate_model():   
    
    processed_data_path=os.path.join(current_path, "processed_data")
    train_data = np.load(os.path.join(processed_data_path, 'train.npz'))
    train_labels = train_data['labels']
    train_data = train_data['data']
    #train_data = train_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    print("train_data shape:", train_data.shape)

    
    valid_data = np.load(os.path.join(processed_data_path, 'val.npz'))
    valid_labels = valid_data['labels']
    valid_data = valid_data['data']
    #valid_data = valid_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    print("valid_data shape:", valid_data.shape)
    
    test_data = np.load(os.path.join(processed_data_path, 'test.npz'))
    test_labels = test_data['labels']
    test_data = test_data['data']
    #test_data = test_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    print("test_data shape:", test_data.shape)

    model_dir =os.path.join(current_path, "model")
    
    
    model_name = f"gesture_fans_{dense_hidden_units}_{dropout_rate}_{LSTM_units}"
    
    for e in range(10,150+1,10):
        
        path = os.path.join(model_dir, f"{model_name}_ep_{e}.weights.h5")
        
        m_structure=model_struct(LSTM_units, dense_hidden_units, dropout_rate, num_classes=5, time_steps=32, width=32, height=32)
        m_structure.load_weights(path)
        m_structure.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        print(f"\nevaluate model : {path}")
    
def load_and_predict(model, data):
    print("data shape: ", data.shape)  # (32, 32)
    data=np.expand_dims(data, axis=0)  # Expand dimensions to match model input shape (1, 32, 32)
    data=np.expand_dims(data, axis=-1) # Expand dimensions to match model input shape (1, 32, 32, 1)
    print("model inferring")
    predictions = model.predict(data)
    probs= predictions[0]*100
    probs_str = [f"{p:.3f}%" for p in probs]
    print("predicted probabilities: ", probs_str)
    
    
evaluate_model()