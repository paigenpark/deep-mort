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

# ==============================================================================
# gaussian NLL loss function (Lakshiminarayanan et al., 2017)
# ==============================================================================
def gaussian_nll_loss(y_true, y_pred):
    """
    Gaussian negative log-likelihood loss for deep ensembles.
    
    The model outputs a 2-unit vector: [mu, raw_variance].
    We split the prediction, apply softplus to get variance > 0,
    and compute:  loss = log(sigma^2)/2 + (y - mu)^2 / (2 * sigma^2)
    
    Args:
        y_true: Ground truth values, shape (batch, 1)
        y_pred: Model predictions, shape (batch, 2) where
                y_pred[:, 0] = mu (predicted mean)
                y_pred[:, 1] = raw_variance (before softplus)
    """
    mu = y_pred[:, 0:1]           # predicted mean
    raw_var = y_pred[:, 1:2]      # raw variance (pre-softplus)
    
    # Softplus to enforce positivity + small floor for numerical stability
    variance = tf.math.softplus(raw_var) + 1e-6
    
    # Gaussian NLL: log(sigma^2)/2 + (y - mu)^2 / (2 * sigma^2)
    nll = 0.5 * tf.math.log(variance) + 0.5 * tf.square(y_true - mu) / variance
    
    return tf.reduce_mean(nll)

def naive_error_loss(y_true, y_pred):
    mu = y_pred[:, 0:1]           # predicted mean
    pred_var = y_pred[:, 1:2]      # raw variance (pre-softplus)
    
    mu_error = (y_true - mu)**2
    mu_loss = tf.reduce_mean(mu_error)

    var_loss = tf.reduce_mean((pred_var - mu_loss)**2)
    total_loss = 0.5 * mu_loss + 0.5 * var_loss
     
    return total_loss

def create_ensemble_model(geo_dim):
    """
    Deep ensemble model for raw mortality rates.
    
    Same architecture as original create_model(), but with two changes:
    1. Output layer has 2 units: [mu, raw_variance] instead of 1
    2. mu uses sigmoid activation (rates are in [0,1]), variance head is linear
    3. Trained with Gaussian NLL instead of MSE
    """
    # defining inputs (identical to original)
    year = tfkl.Input(shape=(1,), dtype='float32', name='Year')
    age =  tfkl.Input(shape=(1,), dtype='int32', name='Age')
    geography = tfkl.Input(shape=(1,), dtype='int32', name='Geography')
    gender = tfkl.Input(shape=(1,), dtype='int32', name='Gender')

    # defining embedding layers (identical to original)
    age_embed = tfkl.Embedding(input_dim=100, output_dim=5, name='Age_embed')(age)
    age_embed = tfkl.Flatten()(age_embed)

    gender_embed = tfkl.Embedding(input_dim=2, output_dim=5, name='Gender_embed')(gender)
    gender_embed = tfkl.Flatten()(gender_embed)

    geography_embed = tfkl.Embedding(input_dim=geo_dim, output_dim=5, name='Geography_embed')(geography)
    geography_embed = tfkl.Flatten()(geography_embed)

    # create feature vector (identical to original)
    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

    # middle layers (identical to original)
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

    # skip connection + final hidden layer (identical to original)
    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    # ---- KEY CHANGE: Two output heads instead of one ----
    # Mean head: sigmoid keeps predictions in [0, 1] for mortality rates
    mu = tfkl.Dense(1, activation='sigmoid', name='mu')(x)
    
    # Variance head: linear output, softplus applied inside the loss function
    raw_var = tfkl.Dense(1, activation='linear', name='raw_variance')(x)
    
    # Concatenate into single output tensor of shape (batch, 2)
    output = tfkl.Concatenate(name='mu_var')([mu, raw_var])

    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[output])
    model.compile(loss=gaussian_nll_loss, optimizer='adam')

    return model


def create_ensemble_log_model(geo_dim):
    """
    Deep ensemble model for log mortality rates.
    
    Same as create_ensemble_model but mu has linear activation (log rates 
    are unbounded) instead of sigmoid.
    """
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

    # create feature vector
    x = tfkl.Concatenate()([year, age_embed, gender_embed, geography_embed])
    x1 = x

    # middle layers
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

    # skip connection + final hidden layer
    x = tfkl.Concatenate()([x1, x])
    x = tfkl.Dense(128, activation='tanh')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(0.05)(x)

    # ---- Two output heads ----
    # Mean head: linear for log-rates (unbounded)
    mu = tfkl.Dense(1, activation='linear', name='mu')(x)
    
    # Variance head: linear output, softplus applied inside the loss
    raw_var = tfkl.Dense(1, activation='linear', name='raw_variance')(x)
    
    output = tfkl.Concatenate(name='mu_var')([mu, raw_var])

    model = tf.keras.Model(inputs=[year, age, geography, gender], outputs=[output])
    model.compile(loss=gaussian_nll_loss, optimizer='adam')

    return model

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def run_deep_ensemble_model(dataset_train, dataset_test, geo_dim, epochs, steps_per_epoch, lograte=False):
    """
    Train a single deep ensemble member.
    
    This is a drop-in replacement for run_deep_model(). Call it M times
    (with different random seeds / initializations) to build your ensemble.
    
    Returns:
        model: trained model that outputs (mu, raw_variance) concatenated
        val_loss: best validation NLL
    """
    if lograte:
        model = create_ensemble_log_model(geo_dim)
    else:
        model = create_ensemble_model(geo_dim)

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.25, patience=3, verbose=0, 
        mode="auto", min_delta=1e-8, cooldown=0, min_lr=0.0
    )]
    
    history = model.fit(
        dataset_train, validation_data=dataset_test, validation_steps=25, 
        steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2, callbacks=callbacks
    )

    val_loss = min(history.history['val_loss'])
    tf.keras.backend.clear_session()

    return model, val_loss


# ============================================================================
# PREDICTION & ENSEMBLE COMBINATION
# ============================================================================

def predict_single_model(model, dataset):
    """
    Get mu and variance predictions from a single ensemble member.
    
    Args:
        model: a trained ensemble model
        dataset: tf.data.Dataset of (features, targets)
    
    Returns:
        mu: predicted means, shape (N, 1)
        variance: predicted variances, shape (N, 1)
    """
    raw_preds = model.predict(dataset)   # shape (N, 2)
    mu = raw_preds[:, 0:1]
    raw_var = raw_preds[:, 1:2]
    
    # Apply same softplus + floor as in the loss function
    variance = np.log(1.0 + np.exp(raw_var)) + 1e-6
    
    return mu, variance


def combine_ensemble_predictions(models, dataset):
    """
    Combine predictions from M ensemble members using the mixture formulas
    from Lakshminarayanan et al. 2017.
    
    The ensemble prediction is a Gaussian with:
        mu* = (1/M) * sum(mu_m)
        sigma*^2 = (1/M) * sum(sigma_m^2 + mu_m^2) - mu*^2
    
    This variance naturally decomposes into:
        - mean of individual variances (avg aleatoric uncertainty)
        - variance of individual means (epistemic uncertainty / model disagreement)
    
    Args:
        models: list of M trained ensemble models
        dataset: tf.data.Dataset of (features, targets)
    
    Returns:
        ensemble_mu: combined mean prediction, shape (N, 1)
        ensemble_var: combined variance (total uncertainty), shape (N, 1)
        aleatoric_var: average predicted variance across members, shape (N, 1)
        epistemic_var: variance of the means across members, shape (N, 1)
    """
    M = len(models)
    all_mus = []
    all_vars = []
    
    for model in models:
        mu, var = predict_single_model(model, dataset)
        all_mus.append(mu)
        all_vars.append(var)
    
    all_mus = np.array(all_mus)      # shape (M, N, 1)
    all_vars = np.array(all_vars)    # shape (M, N, 1)
    
    # Mixture mean: average of individual means
    ensemble_mu = np.mean(all_mus, axis=0)                          # (N, 1)
    
    # Mixture variance (from paper): (1/M) * sum(sigma_m^2 + mu_m^2) - mu*^2
    ensemble_var = np.mean(all_vars + all_mus**2, axis=0) - ensemble_mu**2  # (N, 1)
    
    # Decomposition into aleatoric and epistemic components
    aleatoric_var = np.mean(all_vars, axis=0)                       # (N, 1)
    epistemic_var = np.mean(all_mus**2, axis=0) - ensemble_mu**2    # (N, 1)
    
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
