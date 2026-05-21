import tensorflow as tf
import numpy as np
tfkl = tf.keras.layers

from training_functions import prep_data, get_data, create_model, create_log_model
from code.uncertainty_models.training_functions_uncertainty import (
    gaussian_nll_loss,
    predict_single_model,
    combine_ensemble_predictions,
    get_prediction_intervals,
)


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
