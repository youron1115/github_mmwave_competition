import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback


device = None
if tf.config.list_physical_devices('GPU'):
    device = tf.config.list_physical_devices('GPU')[0]
else:
    device = tf.config.list_physical_devices('CPU')[0]
print(f"Using device: {device}")

current_path = os.path.dirname(os.path.abspath(__file__))

class CustomCheckpoint(Callback):
    def __init__(self, save_path_base, name, save_every_n_epoch=10):
        super().__init__()
        self.save_every_n_epoch = save_every_n_epoch
        self.save_path_base = save_path_base
        self.name = name
        os.makedirs(save_path_base, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.save_every_n_epoch == 0:
            save_path = os.path.join(self.save_path_base, f"{self.name}_ep_{epoch+1}.weights.h5")
            self.model.save_weights(save_path)
            print(f"\nSaved weight at: {save_path}")

def fit_model():   
    
    processed_data_path=os.path.join(current_path, "processed_data")
    """
    print("data path:", processed_data_path)
    exit()
    """
    train_data = np.load(os.path.join(processed_data_path, 'train.npz'))
    #print("key :", train_data.files)
    train_labels = train_data['labels']
    train_data = train_data['data']
    #train_data = train_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    
    #gemini加的
    train_data = np.expand_dims(train_data, axis=-1) # Shape: (N, 20, 32, 32, 1)
    print("train_data shape:", train_data.shape)

    
    valid_data = np.load(os.path.join(processed_data_path, 'val.npz'))
    valid_labels = valid_data['labels']
    valid_data = valid_data['data']
    #valid_data = valid_data.reshape((-1, 32, 32, 1))  # Reshape to (samples, height, width, channels)
    
    #gemini加的
    valid_data = np.expand_dims(valid_data, axis=-1) # Shape: (M, 20, 32, 32, 1)
    print("valid_data shape:", valid_data.shape)
    
    time_steps = 20
    height = 32
    width = 32
    input_shape = (time_steps, width, height, 1)
    num_classes= 4 
    
    dense_hidden_units = 64
    epochs = 100
    LSTM_units = 64
    dropout_rate=0.2
    
    model = tf.keras.Sequential([
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
    
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model_dir =os.path.join(current_path, "model")
    model_name = f"gesture_fans_{dense_hidden_units}_{dropout_rate}_{LSTM_units}"
    
    custom_checkpoint= CustomCheckpoint(
        save_path_base=model_dir,
        save_every_n_epoch=10,
        name=model_name
    )
    
    model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), 
              epochs=epochs, 
              batch_size=200, 
              shuffle=True,
              callbacks=[custom_checkpoint])
    
    print("\nTraining complete")
    
fit_model()
