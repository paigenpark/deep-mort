import tensorflow as tf
import numpy as np
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

# ==============================================================================
# beta NLL loss function (Seitzer et al. 2022)
# ==============================================================================
def beta_nll_loss(y_true, y_pred, beta=0.5):
    mu = y_pred[:, 0:1]
    raw_var = y_pred[:, 1:2]
    
    variance = tf.math.softplus(raw_var) + 1e-6
    
    nll = 0.5 * tf.math.log(variance) + 0.5 * tf.square(y_true - mu) / variance
    
    if beta > 0:
        # Stop gradient so the weighting acts as an adaptive learning rate,
        # not as a term the optimizer differentiates through
        nll = nll * tf.stop_gradient(variance) ** beta
    
    return tf.reduce_mean(nll)

# ==============================================================================
# Freeze-and-train model creation
# ==============================================================================

def create_freeze_model(base_model):
    """
    Take a trained base model from create_model()/create_log_model(),
    freeze all existing layers, and add a variance head branching from
    the last hidden layer.

    The resulting model outputs shape (batch, 2) = [mu, raw_variance],
    identical to the joint-training ensemble format.
    """
    # Freeze all existing layers
    for layer in base_model.layers:
        layer.trainable = False

    # Get the tensor feeding into the final output layer
    final_layer = base_model.get_layer('final')
    last_hidden_output = final_layer.input

    # Existing mu output (frozen)
    mu_output = final_layer.output

    # New trainable variance head
    raw_var = tfkl.Dense(1, activation='linear', name='raw_variance')(last_hidden_output)

    # Concatenate into (batch, 2) to match ensemble output format
    output = tfkl.Concatenate(name='mu_var')([mu_output, raw_var])

    freeze_model = tf.keras.Model(inputs=base_model.inputs, outputs=[output])
    freeze_model.compile(loss=gaussian_nll_loss, optimizer='adam')

    return freeze_model

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


# ============================================================================
# POST-HOC RECALIBRATION (Kuleshov et al., 2018)
# ============================================================================

def fit_recalibration(actual, mu, std, n_quantiles=99):
    """
    Learn a recalibration mapping from a calibration dataset.

    For each of n_quantiles evenly-spaced predicted confidence levels p,
    compute the empirical frequency that actual <= predicted p-th quantile.
    Fit isotonic regression to get a monotone mapping from predicted to
    empirical confidence levels.

    Reference: Kuleshov, Fenner & Ermon (2018), "Accurate Uncertainties for
    Deep Learning Using Calibrated Regression", ICML.

    Args:
        actual:  observed values, shape (N,)
        mu:      predicted means, shape (N,)
        std:     predicted standard deviations, shape (N,)
        n_quantiles: number of quantile levels to evaluate (default 99)

    Returns:
        recal_model: fitted IsotonicRegression (predicted_p -> empirical_p)
        predicted_ps: the quantile grid used, shape (n_quantiles,)
        empirical_ps: observed coverage at each level, shape (n_quantiles,)
    """
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import norm

    predicted_ps = np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)
    empirical_ps = np.empty_like(predicted_ps)

    for i, p in enumerate(predicted_ps):
        # Predicted p-th quantile for each observation
        q_p = mu + norm.ppf(p) * std
        # Fraction of actuals that fall below this quantile
        empirical_ps[i] = np.mean(actual <= q_p)

    recal_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    recal_model.fit(predicted_ps, empirical_ps)

    return recal_model, predicted_ps, empirical_ps


def recalibrated_prediction_intervals(mu, std, recal_model, alpha=0.05):
    """
    Compute recalibrated prediction intervals using a fitted recalibration model.

    Instead of using nominal quantiles (alpha/2 and 1 - alpha/2), find the
    nominal quantile levels whose *empirical* coverage matches the target.

    Args:
        mu:          predicted means, shape (N,)
        std:         predicted standard deviations, shape (N,)
        recal_model: fitted IsotonicRegression from fit_recalibration()
        alpha:       significance level (default 0.05 for 95% interval)

    Returns:
        lower: recalibrated lower bound, shape (N,)
        upper: recalibrated upper bound, shape (N,)
        z_lower: the recalibrated z-score used for the lower bound
        z_upper: the recalibrated z-score used for the upper bound
    """
    from scipy.stats import norm
    from scipy.optimize import brentq

    target_lower = alpha / 2        # 0.025 for 95% PI
    target_upper = 1 - alpha / 2    # 0.975 for 95% PI

    # The recal_model maps: predicted_p -> empirical_p
    # We need the inverse: what predicted_p gives us the desired empirical_p?
    def find_nominal_p(target_empirical):
        """Find predicted_p such that recal_model(predicted_p) = target_empirical."""
        try:
            return brentq(lambda p: recal_model.predict([p])[0] - target_empirical,
                          0.001, 0.999)
        except ValueError:
            # Fallback: if the recalibration curve doesn't span the target,
            # use the nominal level unchanged
            return target_empirical

    nominal_lower = find_nominal_p(target_lower)
    nominal_upper = find_nominal_p(target_upper)

    z_lower = norm.ppf(nominal_lower)
    z_upper = norm.ppf(nominal_upper)

    lower = mu + z_lower * std
    upper = mu + z_upper * std

    return lower, upper, z_lower, z_upper



# ==============================================================================
# Two-phase training
# ==============================================================================

def run_freeze_ensemble_model(dataset_train, dataset_test, geo_dim,
                              epochs_mean, epochs_var, steps_per_epoch,
                              lograte=False):
    """
    Train a single freeze-and-train ensemble member in two phases:

    Phase 1 (MSE): Train the full model for mean prediction using the
        standard architecture from training_functions.
    Phase 2 (Gaussian NLL): Freeze all base layers, add a variance head,
        and train only the variance head.

    Returns:
        model: trained model outputting (mu, raw_variance) concatenated
        val_loss: best validation NLL from Phase 2
    """
    # --- Phase 1: Mean training (MSE loss, all layers trainable) ---
    if lograte:
        base_model = create_log_model(geo_dim)
    else:
        base_model = create_model(geo_dim)

    callbacks_mean = [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.25, patience=3, verbose=0,
        mode="auto", min_delta=1e-8, cooldown=0, min_lr=0.0
    )]

    base_model.fit(
        dataset_train, validation_data=dataset_test, validation_steps=25,
        steps_per_epoch=steps_per_epoch, epochs=epochs_mean, verbose=2,
        callbacks=callbacks_mean
    )

    # --- Phase 2: Variance training (NLL loss, base frozen) ---
    model = create_freeze_model(base_model)

    callbacks_var = [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.25, patience=3, verbose=0,
        mode="auto", min_delta=1e-8, cooldown=0, min_lr=0.0
    )]

    history = model.fit(
        dataset_train, validation_data=dataset_test, validation_steps=25,
        steps_per_epoch=steps_per_epoch, epochs=epochs_var, verbose=2,
        callbacks=callbacks_var
    )

    val_loss = min(history.history['val_loss'])
    tf.keras.backend.clear_session()

    return model, val_loss
