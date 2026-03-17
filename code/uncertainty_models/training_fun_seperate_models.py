import tensorflow as tf
import csv
import numpy as np
import os as os
import matplotlib.pyplot as plt
tfkl = tf.keras.layers

# ==============================================================================
# get and prepare data
# ==============================================================================
def get_data(index, data, max_val, mode, changeratetolog=False):
    if mode == "train":
        rand_index = tf.random.uniform([], minval=0, maxval=max_val, dtype=tf.int32)
        entry = data[rand_index, :]
    elif mode == "not_random":
        entry = data[index, :]
    else:
        rand_index = tf.random.uniform([], minval=0, maxval=max_val, dtype=tf.int32)
        entry = data[rand_index, :]

    geography, gender, year, age, rate = entry[0], entry[1], entry[2], entry[3], entry[4]

    year = (year - 1959) / 60
    age = tf.cast(age, tf.int32)
    geography = tf.cast(geography, tf.int32)
    gender = tf.cast(gender, tf.int32)
    if changeratetolog:
        epsilon = 9e-06
        rate = tf.math.log(tf.maximum(rate, epsilon))

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

    dataset = dataset.batch(256)
    final_data = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return final_data


def prep_error_data(data, squared_errors, mode):
    """
    Prepare a dataset where the target is the squared error from the mean model
    instead of the original rate.

    Args:
        data: original data array with columns [geography, gender, year, age, rate]
        squared_errors: array of squared errors from the mean model, shape (N,)
        mode: "train" or "test"
    """
    # Replace the rate column with squared errors
    data_with_errors = data.copy()
    data_with_errors[:, 4] = squared_errors

    return prep_data(data_with_errors, mode, changeratetolog=False)


# ==============================================================================
# Model 1: Mean model (standard MSE-trained model)
# ==============================================================================
def create_mean_model(geo_dim):
    """
    Standard MLP that predicts mortality rates, trained with MSE.
    Identical to the original create_model() from training_functions.py.
    """
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age =  tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')
    gender = tfkl.Input(shape=(1,), dtype='int32', name='Gender')

    age_embed = tfkl.Embedding(input_dim=100, output_dim=5, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    gender_embed = tfkl.Embedding(input_dim=2, output_dim=5, name='Gender_embed')(gender)
    gender_embed = tfkl.Flatten()(gender_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=5, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

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

    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(1, activation='sigmoid', name='final')(x)

    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[x])
    model.compile(loss='mse', optimizer='adam')

    return model


def create_mean_log_model(geo_dim):
    """
    Standard MLP that predicts log mortality rates, trained with MSE.
    Identical to the original create_log_model() from training_functions.py.
    """
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age =  tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')
    gender = tfkl.Input(shape=(1,), dtype='int32', name='Gender')

    age_embed = tfkl.Embedding(input_dim=100, output_dim=5, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    gender_embed = tfkl.Embedding(input_dim=2, output_dim=5, name='Gender_embed')(gender)
    gender_embed = tfkl.Flatten()(gender_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=5, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

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

    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(1, name='final')(x)

    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[x])
    model.compile(loss='mse', optimizer='adam')

    return model


# ==============================================================================
# Model 2: Error model (predicts squared error of the mean model)
# ==============================================================================
def create_error_model(geo_dim):
    """
    MLP that predicts the squared error of the mean model.
    Same architecture as the mean model, but uses ReLU output (errors >= 0)
    and is trained on squared residuals from Model 1.
    """
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age =  tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')
    gender = tfkl.Input(shape=(1,), dtype='int32', name='Gender')

    age_embed = tfkl.Embedding(input_dim=100, output_dim=5, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    gender_embed = tfkl.Embedding(input_dim=2, output_dim=5, name='Gender_embed')(gender)
    gender_embed = tfkl.Flatten()(gender_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=5, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

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

    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    # ReLU output ensures predicted variance is non-negative
    x = tfkl.Dense(1, activation='relu', name='predicted_variance')(x)

    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[x])
    model.compile(loss='mse', optimizer='adam')

    return model


def create_error_log_model(geo_dim):
    """
    Error model variant for log mortality rates.
    Same as create_error_model but for use when the mean model predicts log-rates.
    """
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age =  tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')
    gender = tfkl.Input(shape=(1,), dtype='int32', name='Gender')

    age_embed = tfkl.Embedding(input_dim=100, output_dim=5, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    gender_embed = tfkl.Embedding(input_dim=2, output_dim=5, name='Gender_embed')(gender)
    gender_embed = tfkl.Flatten()(gender_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=5, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

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

    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    x = tfkl.Dense(1, activation='relu', name='predicted_variance')(x)

    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[x])
    model.compile(loss='mse', optimizer='adam')

    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_squared_errors(mean_model, data, changeratetolog=False):
    """
    Compute squared errors of the mean model on the given data.

    Args:
        mean_model: trained Model 1
        data: raw data array with columns [geography, gender, year, age, rate]
        changeratetolog: whether rates were log-transformed for training

    Returns:
        squared_errors: array of shape (N,) with squared errors per sample
    """
    dataset = prep_data(data, mode="not_random", changeratetolog=changeratetolog)

    predictions = mean_model.predict(dataset)  # shape (N, 1)

    # Get the true targets in the same order (not_random mode preserves order)
    rates = data[:, 4].copy()
    if changeratetolog:
        epsilon = 9e-06
        rates = np.log(np.maximum(rates, epsilon))

    predictions = predictions[:len(rates)].flatten()
    squared_errors = (rates - predictions) ** 2

    return squared_errors


def run_two_stage_model(dataset_train, dataset_test, train_data, geo_dim,
                        epochs, steps_per_epoch, lograte=False):
    """
    Train the two-stage uncertainty model:
      Stage 1: Train mean model (MSE loss) to predict mortality rates
      Stage 2: Train error model (MSE loss) to predict squared errors from Stage 1

    Args:
        dataset_train: tf.data.Dataset for training (features, rates)
        dataset_test: tf.data.Dataset for validation (features, rates)
        train_data: raw training data array (needed to compute squared errors)
        geo_dim: number of unique geographies
        epochs: number of training epochs per model
        steps_per_epoch: training steps per epoch
        lograte: whether to use log-rate models

    Returns:
        mean_model: trained Model 1 (predicts mean)
        error_model: trained Model 2 (predicts squared error / variance)
        mean_val_loss: best validation loss for Model 1
        error_val_loss: best validation loss for Model 2
    """
    # --- Stage 1: Train mean model ---
    if lograte:
        mean_model = create_mean_log_model(geo_dim)
    else:
        mean_model = create_mean_model(geo_dim)

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.25, patience=3, verbose=0,
        mode="auto", min_delta=1e-8, cooldown=0, min_lr=0.0
    )]

    print("=== Stage 1: Training mean model ===")
    history_mean = mean_model.fit(
        dataset_train, validation_data=dataset_test, validation_steps=25,
        steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2, callbacks=callbacks
    )
    mean_val_loss = min(history_mean.history['val_loss'])

    # --- Stage 2: Compute squared errors and train error model ---
    print("=== Computing squared errors from mean model ===")
    squared_errors = compute_squared_errors(mean_model, train_data, changeratetolog=lograte)

    # Build training dataset where target = squared error
    error_train = prep_error_data(train_data, squared_errors, mode="train")

    # For validation, compute squared errors on validation data too
    # (we reuse dataset_test features but need error targets)
    # Use dataset_test as-is for validation_data structure, but the error model
    # validates against its own error predictions

    if lograte:
        error_model = create_error_log_model(geo_dim)
    else:
        error_model = create_error_model(geo_dim)

    callbacks_error = [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.25, patience=3, verbose=0,
        mode="auto", min_delta=1e-8, cooldown=0, min_lr=0.0
    )]

    print("=== Stage 2: Training error model ===")
    history_error = error_model.fit(
        error_train, steps_per_epoch=steps_per_epoch,
        epochs=epochs, verbose=2, callbacks=callbacks_error
    )
    error_val_loss = min(history_error.history['loss'])

    tf.keras.backend.clear_session()

    return mean_model, error_model, mean_val_loss, error_val_loss


# ============================================================================
# PREDICTION & ENSEMBLE COMBINATION
# ============================================================================

def predict_two_stage(mean_model, error_model, dataset):
    """
    Get mean and variance predictions from the two-stage model.

    Args:
        mean_model: trained Model 1
        error_model: trained Model 2
        dataset: tf.data.Dataset of (features, targets)

    Returns:
        mu: predicted means, shape (N, 1)
        variance: predicted variance (squared error), shape (N, 1)
    """
    mu = mean_model.predict(dataset)          # shape (N, 1)
    variance = error_model.predict(dataset)    # shape (N, 1)

    # Floor variance at a small positive value for numerical stability
    variance = np.maximum(variance, 1e-6)

    return mu, variance


def combine_ensemble_predictions(mean_models, error_models, dataset):
    """
    Combine predictions from M ensemble pairs (mean_model, error_model)
    using the same mixture formulas as Lakshminarayanan et al. 2017.

    Uncertainty decomposition:
        - Aleatoric: average predicted variance across ensemble members
        - Epistemic: variance of ensemble means (model disagreement)

    Args:
        mean_models: list of M trained mean models
        error_models: list of M trained error models
        dataset: tf.data.Dataset of (features, targets)

    Returns:
        ensemble_mu: combined mean prediction, shape (N, 1)
        ensemble_var: combined variance (total uncertainty), shape (N, 1)
        aleatoric_var: average predicted variance across members, shape (N, 1)
        epistemic_var: variance of the means across members, shape (N, 1)
    """
    M = len(mean_models)
    all_mus = []
    all_vars = []

    for mean_model, error_model in zip(mean_models, error_models):
        mu, var = predict_two_stage(mean_model, error_model, dataset)
        all_mus.append(mu)
        all_vars.append(var)

    all_mus = np.array(all_mus)      # shape (M, N, 1)
    all_vars = np.array(all_vars)    # shape (M, N, 1)

    # Mixture mean
    ensemble_mu = np.mean(all_mus, axis=0)

    # Mixture variance: (1/M) * sum(sigma_m^2 + mu_m^2) - mu*^2
    ensemble_var = np.mean(all_vars + all_mus**2, axis=0) - ensemble_mu**2

    # Decomposition
    aleatoric_var = np.mean(all_vars, axis=0)
    epistemic_var = np.mean(all_mus**2, axis=0) - ensemble_mu**2

    return ensemble_mu, ensemble_var, aleatoric_var, epistemic_var


# ============================================================================
# CONVENIENCE: Get uncertainty intervals
# ============================================================================

def get_prediction_intervals(ensemble_mu, ensemble_var, z=1.96):
    """
    Compute prediction intervals from ensemble mean and variance.

    Args:
        ensemble_mu: ensemble mean predictions, shape (N, 1)
        ensemble_var: ensemble variance predictions, shape (N, 1)
        z: z-score for the interval (default 1.96 = 95% interval)

    Returns:
        lower: lower bound of prediction interval
        upper: upper bound of prediction interval
    """
    ensemble_std = np.sqrt(ensemble_var)
    lower = ensemble_mu - z * ensemble_std
    upper = ensemble_mu + z * ensemble_std
    return lower, upper
