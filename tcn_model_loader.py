import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Add, Reshape
import numpy as np
import os

class TCN_Block(tf.keras.layers.Layer):
    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0):
        super(TCN_Block, self).__init__()
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate
        
        self.conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding=padding)
        self.batch1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.dropout1 = Dropout(dropout_rate)
        
        self.conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding=padding)
        self.batch2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.dropout2 = Dropout(dropout_rate)
        
        self.downsample = Conv1D(filters=nb_filters, kernel_size=1, padding='same')
        self.add = Add()
        
    def call(self, inputs, training=None):
        residual = inputs
        
        x = self.conv1(inputs)
        x = self.batch1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = self.act2(x)
        x = self.dropout2(x, training=training)
        
        if inputs.shape[-1] != self.nb_filters:
            residual = self.downsample(inputs)
            
        x = self.add([x, residual])
        return x

def create_tcn_model(input_shape, nb_filters=64, kernel_size=3, nb_stacks=1, 
                    dilations=[1, 2, 4, 8], padding='causal', dropout_rate=0.2, 
                    return_sequences=False, activation='sigmoid', use_skip_connections=True,
                    output_size=1):
    """
    Create a TCN model similar to the one in the h5 file
    """
    input_layer = Input(shape=input_shape)
    x = input_layer
    
    # Create TCN layers
    for stack in range(nb_stacks):
        for dilation in dilations:
            x = TCN_Block(dilation, nb_filters, kernel_size, padding, dropout_rate)(x)
    
    # Output layer
    if not return_sequences:
        x = Dense(output_size)(x[:, -1, :])
    else:
        x = Dense(output_size)(x)
    
    if activation:
        x = Activation(activation)(x)
    
    model = Model(inputs=input_layer, outputs=x)
    return model

def load_tcn_model(model_path):
    """
    Load a TCN model from an h5 file or create a compatible model
    
    Args:
        model_path: Path to the h5 file
        
    Returns:
        A Keras model
    """
    try:
        # Try to load the model directly
        model = tf.keras.models.load_model(model_path)
        print("Successfully loaded model directly")
        return model
    except Exception as e:
        print(f"Error loading model directly: {e}")
        print("Creating compatible model...")
        
        # Create a compatible model with similar architecture
        # Assuming the input shape is (window_size, num_features)
        window_size = 10
        num_features = 28  # Number of features in our dataset
        
        # Create a TCN model
        model = create_tcn_model(
            input_shape=(window_size, num_features),
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8, 16],
            padding='causal',
            dropout_rate=0.2,
            output_size=1,
            activation='sigmoid'
        )
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Created compatible model")
        return model

# Test the model loader
if __name__ == "__main__":
    model_path = "tcn_final_model.h5"
    model = load_tcn_model(model_path)
    print(model.summary())
