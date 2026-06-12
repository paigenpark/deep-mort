import tensorflow as tf
import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt
tfkl = tf.keras.layers

# get and prepare data
def get_data(index, data, max_val, mode, changeratetolog=False):
    if mode == "train":
        # Randomly selects index from training data between 0 and the max index in train
        rand_index = tf.random.uniform([], minval=0, maxval=max_val, dtype=tf.int32) 
        entry = data[rand_index, :]
    elif mode == "not_random":
        # Selects specified index from test data 
        entry = data[index, :]
    else:  # Assuming mode="test" or any other value
        # For any other value of mode, randomly selects index from test
        rand_index = tf.random.uniform([], minval=0, maxval=max_val, dtype=tf.int32)
        entry = data[rand_index, :]

    geography, gender, year, age, rate = entry[0], entry[1], entry[2], entry[3], entry[4]

    # Normalization or preparation
    year = (year - 1959) / 60
    age = tf.cast(age, tf.int32)
    geography = tf.cast(geography, tf.int32)
    gender = tf.cast(gender, tf.int32)
    if changeratetolog:
        epsilon = 9e-06 # min rate in training data
        rate = tf.math.log(tf.maximum(rate, epsilon))

    # Reshape each element to scalar
    features = (tf.reshape(year, [1]), tf.reshape(age, [1]), 
                tf.reshape(geography, [1]), tf.reshape(gender, [1]))
    rate = tf.reshape(rate, [1])
    return features, rate

    
def prep_data(data, mode, changeratetolog=False):
    
    data = tf.convert_to_tensor(data)
    data = tf.cast(data, tf.float32)
    max_val = data.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices(np.arange(10000))

    if mode == "train":
        dataset = dataset.repeat()
    
    else:
        dataset = dataset.repeat(120)
    
    dataset = dataset.map(
        lambda x: get_data(x, data, max_val=max_val, mode=mode, changeratetolog=changeratetolog), 
                          num_parallel_calls=4)

    # Batch the dataset for efficient predictions 
    # Each batch consists of 2 parts - batch of features and batch of targets
    dataset = dataset.batch(256)

    # Prefetch to improve performance
    final_data = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return final_data

# create DL model
def create_model(geo_dim):
    # defining inputs 
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age =  tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')
    gender = tfkl.Input(shape=(1,), dtype='int32', name='Gender')

    # defining embedding layers 
    age_embed = tfkl.Embedding(input_dim=100, output_dim=5, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    gender_embed = tfkl.Embedding(input_dim=2, output_dim=5, name='Gender_embed')(gender)
    gender_embed = tfkl.Flatten()(gender_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=5, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    # create feature vector that concatenates all inputs 
    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

    # setting up middle layers 
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    # setting up output layer 
    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(1, activation='sigmoid', name='final')(x)

    # creating the model 
    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[x])

    # compiling the model
    model.compile(loss='mse', optimizer='adam')

    return model

def create_log_model(geo_dim):
    # defining inputs 
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age =  tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')
    gender = tfkl.Input(shape=(1,), dtype='int32', name='Gender')

    # defining embedding layers 
    age_embed = tfkl.Embedding(input_dim=100, output_dim=5, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    gender_embed = tfkl.Embedding(input_dim=2, output_dim=5, name='Gender_embed')(gender)
    gender_embed = tfkl.Flatten()(gender_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=5, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    # create feature vector that concatenates all inputs 
    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

    # setting up middle layers 
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    # setting up output layer 
    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)
    
    x = tfkl.Dense(1, name='final')(x)

    # creating the model 
    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[x])

    # compiling the model
    model.compile(loss='mse', optimizer='adam')

    return model


# run DL model
def run_deep_model(dataset_train, dataset_test, geo_dim, epochs, steps_per_epoch, lograte=False):
    if lograte:
        model = create_log_model(geo_dim)
    else:
        model = create_model(geo_dim)

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=3, verbose=0, mode="auto", 
                                                    min_delta=1e-8, cooldown=0, min_lr=0.0)]
    history = model.fit(dataset_train, validation_data=dataset_test, validation_steps=25, steps_per_epoch=steps_per_epoch, 
                        epochs=epochs, verbose=2, callbacks=callbacks)

    val_loss = min(history.history['val_loss'])

    tf.keras.backend.clear_session()

    return model, val_loss

