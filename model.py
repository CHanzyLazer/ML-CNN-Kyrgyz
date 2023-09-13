from tensorflow import keras
from keras import layers


def get_model(base_conv_num=32, dropout=False, BN=False, append_layers_1=0, append_layers_2=1, input_shape=(32, 32, 1), output_size=36)->keras.Model:
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Conv2D(base_conv_num, 3, activation="relu"))
    if BN: model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(base_conv_num, 3, activation="relu"))
    if BN: model.add(layers.BatchNormalization())
    for _ in range(append_layers_1):
        model.add(layers.Conv2D(base_conv_num, 3, activation="relu", padding="same"))
        if BN: model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2))
    if dropout: model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(base_conv_num*2, 3, activation="relu"))
    if BN: model.add(layers.BatchNormalization())
    for _ in range(append_layers_2):
        model.add(layers.Conv2D(base_conv_num*2, 3, activation="relu", padding="same"))
        if BN: model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2))
    if dropout: model.add(layers.Dropout(0.5))
    model.add(layers.DepthwiseConv2D(3, strides=3, depth_multiplier=max(min(32//base_conv_num, 8), 1), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_size, activation="softmax"))
    
    return model
