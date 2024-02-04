import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import category_encoders as ce   # https://contrib.scikit-learn.org/category_encoders/
import keras
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping
import random
from sklearn.model_selection import KFold
from tensorflow.keras import models, layers

sys.path.append("../")
from utils import (
    split, set_tf_loglevel,
    evaluate_model, evaluate_predictions, 
    plot_from_model, plot_from_predictions, plot_loss_curves, 
    embedding_preproc,
    plot_ridgeline
)
from glmmnet import DistParams, build_glmmnet, predict_glmmnet, build_baseline_nn
tfd = tfp.distributions

def build_glmmnet_independent(cardinality, num_vars, final_layer_likelihood, train_size, hyperparameters = None, random_state=42, regularizer=False, is_prior_trainable=False):
    """
    Build a GLMMNet model.

    Parameters
    ----------
    cardinality: int, number of unique values in the high-cardinality variable of interest
    num_vars: list of numerical variables
    final_layer_likelihood: str, likelihood of the final layer, "gaussian" or "gamma"
    train_size: int, number of training samples (used in kl_weight)
    random_state: int, random seed
    regularizer: bool, whether to use l2 regulariser, default is False
    is_prior_trainable: bool, whether to train the prior parameters, default is False
    """
    # Set random seed
    tf.random.set_seed(random_state)
    
    # default hyperparameters
    if hyperparameters == None:
        hidden_units = [64, 32, 16]
        regularization_factor = 0.01
        learning_rate = 0.01
    else:
        hidden_units = hyperparameters['hidden_units']
        regularization_factor = hyperparameters['l2_regularization_factor']
        learning_rate = hyperparameters['learning_rate']

    # Construct a standard FFNN for the fixed effects
    num_inputs = layers.Input(shape=(len(num_vars),), name="numeric_inputs")
    hidden_activation = "relu"
    x = num_inputs
    for hidden_layer in range(len(hidden_units)):
        units = hidden_units[hidden_layer]
        x = layers.Dense(
            units=units, activation=hidden_activation, 
            kernel_regularizer=tf.keras.regularizers.l2(regularization_factor) if regularizer else None,
            name=f"hidden_{hidden_layer + 1}")(x)
    f_X = layers.Dense(units=1, activation="linear", name="f_X")(x)

    # Deal with categorical inputs (random effects)
    cat_inputs = layers.Input(shape=(cardinality,), name=f"category_OHE_inputs")
    # Construct the random effects, by variational inference
    # Code adapted from https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression
    # Specify the surrogate posterior over the random effects
    def posterior_u(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer="random_normal"),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n], scale=0.01 * tf.nn.softplus(t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])
    # Specify the prior over the random effects
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(0.1)) # Inverse of softplus()
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(1, dtype=dtype, initializer="random_normal"),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=tf.zeros(n), scale=tf.nn.softplus(c + t)),
                reinterpreted_batch_ndims=1)),
        ])
    def prior_fixed(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n), scale=0.1 * tf.ones(n)), 
            reinterpreted_batch_ndims=1
        )
    RE = tfp.layers.DenseVariational(
        units=1, 
        make_posterior_fn=posterior_u,
        make_prior_fn=prior_trainable if is_prior_trainable else prior_fixed,
        kl_weight=1 / train_size,
        use_bias=False,
        activity_regularizer=tf.keras.regularizers.l2(regularization_factor) if regularizer else None,
        name="RE",)(cat_inputs)

    # Add the RE to the f(X) output
    eta = layers.Add(name="f_X_plus_RE")([f_X, RE])

    # Build the final layer
    if final_layer_likelihood == "gaussian":
        # Compute the distributional parameters
        dist_params = DistParams(inverse_link="identity", phi_init=0.1, name="dist_params",)(eta)
        # Construct the distribution output
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1], scale=t[..., 1:]), 
            name="distribution",
        )(dist_params)
    elif final_layer_likelihood == "gamma":
        # Compute the distributional parameters
        dist_params = DistParams(
            inverse_link="exp",
            phi_init=0.1,
            name="dist_params",)(eta)
        # Construct the distribution output
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Gamma(                    # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Gamma
                concentration=1 / t[..., 1:],       # concentration = shape = 1 / dispersion
                rate=(1 / t[..., 1:]) / t[..., :1], # rate = shape / location
            ), name="distribution")(dist_params)
    
    glmmnet = models.Model(inputs=[num_inputs, cat_inputs], outputs=output)

    def NLL(y_true, y_dist_pred):
        return -y_dist_pred.log_prob(y_true)
    
    glmmnet.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss = NLL)

    return glmmnet

def glmmnet_results(regularizer, n_prediction_samples, likelihood, data, hyperparameters = None, random_state = 42):
    """
    Get results from GLMMNet on training and validation sets
    ------------------------
    regularizer: the type of l2 regularization to be used. If cluster_type == None, regularizer is a boolean. If cluster_type != None, regularizer is boolean or specified by a string (either 'loc' or 'scale').
    When a boolean, True indicates to use l2 regularisation on all parameters. False uses no regularization. 'loc' means to use regularization on the location parameters if assuming dependence between categories.
    'scale' only regularizes the covariance matrix in the posterior distribution.
    cluster_type: None, 'ANZSIC' or an integer. None means to assume independence between categories. ANZSIC uses the ANZSIC classification to cluster dependent blocks of categories. If it is an integer,
    the integer represents the number of clusters to use during KNN. 
    n_prediction_samples: int; the number of samples to average over when making predictions.
    hyperparameters: dict; contains the hidden_units, l2 regularization factor and learning rate. If None, use default parameters that were used initially. check build_glmmnet_dependent or build_glmmnet_independent
    """
    X_train, y_train, X_test, y_test = data
    y_dist = likelihood
    hicard_var = 'category'
    x_num = [col for col in X_train.columns if col not in hicard_var]
    colnames = [hicard_var] + x_num
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    # Convert numeric to string so they can be recognised by ce.OrdinalEncoder
    X_train[hicard_var] = X_train[hicard_var].astype("str")
    X_test[hicard_var] = X_test[hicard_var].astype("str")

    ct_nn = make_column_transformer(
        (ce.ordinal.OrdinalEncoder(handle_unknown="value"), [hicard_var]), # "value" encodes unknown categories as "-1"
        (MinMaxScaler(feature_range=(0, 1)), x_num),
    )

    fix_missing_vales = make_column_transformer(
        (SimpleImputer(missing_values=-1, strategy="constant", fill_value=0), [hicard_var])
    )

    # Fit and transform the data using the ColumnTransformer
    X_train_ct = pd.DataFrame(ct_nn.fit_transform(X_train), columns= colnames)
    X_test_ct = pd.DataFrame(ct_nn.transform(X_test), columns= colnames)
    X_test_ct[hicard_var] = fix_missing_vales.fit_transform(X_test_ct)

    one_hot = ce.OneHotEncoder(cols=[hicard_var], handle_unknown="value")
    X_train_ct_ohe = one_hot.fit_transform(X_train_ct)
    X_test_ct_ohe = one_hot.transform(X_test_ct)

    start = time.time()
    
    cardinality = len(X_train[hicard_var].unique())

    glmmnet = build_glmmnet_independent(cardinality=cardinality, num_vars=x_num, final_layer_likelihood= y_dist, 
                                        train_size=X_train.shape[0], hyperparameters = hyperparameters, random_state=random_state, regularizer = regularizer)

    es = EarlyStopping(patience=50, restore_best_weights=True, monitor="val_loss", verbose=0)

    hist = glmmnet.fit(
        (X_train_ct.drop([hicard_var], axis=1), X_train_ct_ohe.loc[:, X_train_ct_ohe.columns.str.startswith(hicard_var)]), y_train, 
        validation_split=0.2,
        callbacks=[es],
        # note: batch_size = 128 (128 or 256 doesn't make a difference, but 256 is faster)
        batch_size=256, epochs=500, verbose=False)

    y_pred_train = predict_glmmnet(glmmnet, X_train_ct_ohe, hicard_var, n_prediction_samples)
    y_pred_test = predict_glmmnet(glmmnet, X_test_ct_ohe, hicard_var, n_prediction_samples)
    
    end = time.time()
    runtimes = end - start
    sigma_e = glmmnet.get_layer("dist_params").get_weights()

    train_scores = evaluate_predictions(y_train, y_pred_train, categories=X_train[hicard_var], likelihood= y_dist, gamma_shape=1 / sigma_e[0][0])
    test_scores = evaluate_predictions(y_test, y_pred_test, categories=X_test[hicard_var], likelihood= y_dist, gamma_shape=1 / sigma_e[0][0])
                                        
    return runtimes, glmmnet, train_scores, test_scores 

def glmmnet_results_k_fold(exp_id, k, y_dist, regularizer, cluster_type, n_prediction_samples, hyperparameters = None, is_prior_trainable = False, random_state = 42):
    """
    Get results from GLMMNet on training and testing sets
    ------------------------
    regularizer: the type of l2 regularization to be used. If cluster_type == None, regularizer is a boolean. If cluster_type != None, regularizer is boolean or specified by a string (either 'loc' or 'scale').
    When a boolean, True indicates to use l2 regularisation on all parameters. False uses no regularization. 'loc' means to use regularization on the location parameters if assuming dependence between categories.
    'scale' only regularizes the covariance matrix in the posterior distribution.
    cluster_type: None, 'ANZSIC' or an integer. None means to assume independence between categories. ANZSIC uses the ANZSIC classification to cluster dependent blocks of categories. If it is an integer,
    the integer represents the number of clusters to use during KNN. 
    n_prediction_samples: int; the number of samples to average over when making predictions.
    hyperparameters: dict; contains the hidden_units, l2 regularization factor and learning rate. If None, use default parameters that were used initially. check build_glmmnet_dependent or build_glmmnet_independent
    validation: if using validation split (for hyperparameter tuning) 
    """
    all_data = get_all_data(k, exp_id)
    val_scores = []
    for i in range(k):
        print(f"k-fold validation {i+1}/{k}")
        X_train, y_train, X_val, y_val = all_data[i]
        hicard_var = 'category'
        x_num = [col for col in X_train.columns if col not in hicard_var]
        colnames = [hicard_var] + x_num
        
        X_train = X_train.copy()
        X_val = X_val.copy()
        # Convert numeric to string so they can be recognised by ce.OrdinalEncoder
        X_train[hicard_var] = X_train[hicard_var].astype("str")
        X_val[hicard_var] = X_val[hicard_var].astype("str")

        ct_nn = make_column_transformer(
            (ce.ordinal.OrdinalEncoder(handle_unknown="value"), [hicard_var]), # "value" encodes unknown categories as "-1"
            (MinMaxScaler(feature_range=(0, 1)), x_num),
        )

        fix_missing_vales = make_column_transformer(
            (SimpleImputer(missing_values=-1, strategy="constant", fill_value=0), [hicard_var])
        )

        # Fit and transform the data using the ColumnTransformer
        X_train_ct = pd.DataFrame(ct_nn.fit_transform(X_train), columns= colnames)
        X_val_ct = pd.DataFrame(ct_nn.transform(X_val), columns= colnames)
        X_val_ct[hicard_var] = fix_missing_vales.fit_transform(X_val_ct)

        one_hot = ce.OneHotEncoder(cols=[hicard_var], handle_unknown="value")
        X_train_ct_ohe = one_hot.fit_transform(X_train_ct)
        X_val_ct_ohe = one_hot.transform(X_val_ct)
        cardinality = len(X_train[hicard_var].unique())

        glmmnet = build_glmmnet_independent(
            cardinality=cardinality, num_vars=x_num, final_layer_likelihood= y_dist, 
            train_size=X_train.shape[0], hyperparameters = hyperparameters, random_state=random_state, regularizer = regularizer)

        es = EarlyStopping(patience=50, restore_best_weights=True, monitor="val_loss", verbose=0)

        hist = glmmnet.fit(
            (X_train_ct.drop([hicard_var], axis=1), X_train_ct_ohe.loc[:, X_train_ct_ohe.columns.str.startswith(hicard_var)]), y_train, 
            validation_split=0.2,
            callbacks=[es],
            # note: batch_size = 128 (128 or 256 doesn't make a difference, but 256 is faster)
            batch_size=256, epochs=500, verbose=False)
        sigma_e = glmmnet.get_layer("dist_params").get_weights()
        
        X_val_predict = X_val_ct_ohe
            
        y_pred_val = predict_glmmnet(glmmnet, X_val_predict, hicard_var, n_prediction_samples)
        val_scores_i = evaluate_predictions(y_val, y_pred_val, categories=X_val[hicard_var], likelihood = y_dist, gamma_shape=1 / sigma_e[0][0])
        val_scores.append(val_scores_i)

    return {'val_scores': val_scores}

def GLMMNet_hyperparameter_tuning_k_fold(exp_id, k, regularizer, cluster_type, y_dist, n_search, n_prediction_samples):
    tuning_results = []
    hyperparameters_tested = []
    val_scores_all = []
    
    learning_rate_range = 0.1**np.arange(1, 4, 1)  # Range for learning rate
    l2_regularization_range = 0.1**np.arange(1,4, 1)  # Range for l2 regularization factor
    hidden_units_range = 2**np.arange(4,10,2)  # Range for number of units in hidden layers
    n_layers_range = np.arange(2,5)
    
    i = 0
    while i < n_search:
        if i == 0:
            hyperparameters = {'learning_rate': 0.01, 'l2_regularization_factor': 0.01, 'n_layers': 3, 'hidden_units': [64, 32, 16]} # test default parameters
        else:
            # random search of hyperparameters
            learning_rate = random.choice(learning_rate_range) 
            if regularizer == False:
                l2_regularization_factor = 0.01
            else:
                l2_regularization_factor = random.choice(l2_regularization_range)
            n_layers = random.choice(n_layers_range)
            hidden_units = random.choices(hidden_units_range, k = n_layers)
            hyperparameters = {'learning_rate': learning_rate, 'l2_regularization_factor': l2_regularization_factor, 'n_layers': n_layers, 'hidden_units': hidden_units}
        print(hyperparameters)
        if hyperparameters not in hyperparameters_tested:
            hyperparameters_tested.append(hyperparameters)
            try:
                print(f"trying parameter search {i+1} / {n_search}")
                results = glmmnet_results_k_fold(exp_id, k, y_dist, regularizer, cluster_type, n_prediction_samples, hyperparameters)
                val_scores = pd.DataFrame(results["val_scores"])
                mean_val_scores = val_scores.mean()
                val_scores_all.append(val_scores)
                tuning_results.append(mean_val_scores)
                i += 1
            except:
                pass
            
    tuning_results = pd.DataFrame(tuning_results)
    top_hyperparameters = hyperparameters_tested[tuning_results['CRPS'].idxmin()]  
    return top_hyperparameters, tuning_results, hyperparameters_tested, val_scores_all

def build_baseline_nn_hyperparameters(X_train, objective="mse", hyperparameters = None, print_embeddings=False, random_state=42, cat_vars=[], num_vars=[]):
    """
    Build a baseline neural network model with embeddings.
    Code adapted from https://github.com/oegedijk/keras-embeddings/blob/72c1cfa29b1c57b5a14c24781f9dc713becb68ec/build_embeddings.py#L38
    hyperparameters: dict with keys 'hidden_units' (list), 'dropout_rate' and 'learning_rate'.
    """
    tf.random.set_seed(random_state)
    inputs = []
    embeddings = []
    
    # default hyperparameters
    if hyperparameters == None:
        hidden_units = [64, 32, 16]
        dropout_rate = 0.2
        learning_rate = 0.01
    else:
        hidden_units = hyperparameters['hidden_units']
        dropout_rate = hyperparameters['dropout_rate']
        learning_rate = hyperparameters['learning_rate']


    for col in cat_vars:
        # Estimate cardinality on the training set
        cardinality = int(np.ceil(X_train[col].nunique()))
        # Set the embedding dimension
        embedding_dim = int(max(cardinality ** (1/4), 2))
        if print_embeddings:
            print(f'[{col}] cardinality: {cardinality} and embedding dim: {embedding_dim}')
        
        # Construct the embedding layer
        col_inputs = layers.Input(shape=(1, ), name=col+"_input")
        embedding = layers.Embedding(input_dim=cardinality + 1, output_dim=embedding_dim, input_length=1, name=col+"_embed")(col_inputs)
        # Use SpatialDropout to prevent overfitting
        # See: https://stackoverflow.com/questions/50393666/how-to-understand-spatialdropout1d-and-when-to-use-it
        embedding = layers.SpatialDropout1D(dropout_rate, name=col+"dropout")(embedding)
        # Flatten out the embeddings
        embedding = layers.Reshape(target_shape=(embedding_dim,), name=col)(embedding)
        # Add the input shape to inputs
        inputs.append(col_inputs)
        # Add the embeddings to the embeddings layer
        embeddings.append(embedding)

    # Add numeric inputs
    num_inputs = layers.Input(shape=(len(num_vars),), name="numeric_inputs")
    inputs.append(num_inputs)

    # Paste all the inputs together
    x = layers.Concatenate(name="combined_inputs")(embeddings + [num_inputs])

    # Add some general NN layers
    hidden_activation = "relu"
    output_activation = "linear"
    for hidden_layer in range(len(hidden_units)):
        units = hidden_units[hidden_layer]
        x = layers.Dense(units=units, activation=hidden_activation, name=f"hidden_{hidden_layer + 1}")(x)
    output = layers.Dense(units=1, activation=output_activation, name="output")(x)
    model = models.Model(inputs=inputs, outputs=output)

    # Compile the model
    # model.compile(optimizer="adam", loss=objective, metrics=["mae"])
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss = objective) 

    return model

def NN_ee_results(data, likelihood, hyperparameters = None, random_state = 42):
    X_train, y_train, X_test, y_test = data
    
    y_dist = likelihood
    hicard_var = 'category'
    x_num = [col for col in X_train.columns if col not in hicard_var]
    colnames = [hicard_var] + x_num

    # Convert numeric to string so they can be recognised by ce.OrdinalEncoder
    X_train[hicard_var] = X_train[hicard_var].astype("str")
    X_test[hicard_var] = X_test[hicard_var].astype("str")

    ct_nn = make_column_transformer(
        (ce.ordinal.OrdinalEncoder(handle_unknown="value"), [hicard_var]), # "value" encodes unknown categories as "-1"
        (MinMaxScaler(feature_range=(0, 1)), x_num),
    )

    fix_missing_vales = make_column_transformer(
        (SimpleImputer(missing_values=-1, strategy="constant", fill_value=0), [hicard_var])
    )

    # Fit and transform the data using the ColumnTransformer
    X_train_ct = pd.DataFrame(ct_nn.fit_transform(X_train), columns= colnames)
    X_test_ct = pd.DataFrame(ct_nn.transform(X_test), columns= colnames)
    X_test_ct[hicard_var] = fix_missing_vales.fit_transform(X_test_ct)

    one_hot = ce.OneHotEncoder(cols=[hicard_var], handle_unknown="value")
    X_train_ct_ohe = one_hot.fit_transform(X_train_ct)
    X_test_ct_ohe = one_hot.transform(X_test_ct)
    
    X_embed_train, X_embed_test = embedding_preproc(X_train_ct, X_test_ct, [hicard_var])
    
    cardinality = len(X_train[hicard_var].unique()) 
    start = time.time()

    # Define training parameters
    epochs = 500
    patience = 50
    batch_size = 256
    es = EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss", verbose=2)

    NN_ee = build_baseline_nn_hyperparameters(X_train_ct, hyperparameters = hyperparameters, random_state=random_state, num_vars=x_num, cat_vars=[hicard_var])
    NN_ee.fit(
        (tuple(X_embed_train), X_train_ct.drop([hicard_var], axis=1)), y_train, validation_split=0.2, 
        epochs=epochs, callbacks=[es], batch_size=batch_size, verbose=0)

    y_pred_train = NN_ee.predict((tuple(X_embed_train), X_train_ct.drop([hicard_var], axis=1))).flatten()
    y_pred_test = NN_ee.predict((tuple(X_embed_test), X_test_ct.drop([hicard_var], axis=1))).flatten()

    train_scores = evaluate_predictions(y_train, y_pred_train, categories=X_train[hicard_var], likelihood= y_dist)
    test_scores = evaluate_predictions(y_test, y_pred_test, categories=X_test[hicard_var], likelihood=y_dist)

    end = time.time()
    runtimes = end - start
    extra_params = 0

    return runtimes, extra_params, NN_ee, train_scores, test_scores

def NN_ee_results_k_fold(exp_id, k, likelihood, hyperparameters = None, random_state = 42):
    """
    Get results from GLMMNet on training and testing sets
    ------------------------
    regularizer: the type of l2 regularization to be used. If cluster_type == None, regularizer is a boolean. If cluster_type != None, regularizer is boolean or specified by a string (either 'loc' or 'scale').
    When a boolean, True indicates to use l2 regularisation on all parameters. False uses no regularization. 'loc' means to use regularization on the location parameters if assuming dependence between categories.
    'scale' only regularizes the covariance matrix in the posterior distribution.
    cluster_type: None, 'ANZSIC' or an integer. None means to assume independence between categories. ANZSIC uses the ANZSIC classification to cluster dependent blocks of categories. If it is an integer,
    the integer represents the number of clusters to use during KNN. 
    n_prediction_samples: int; the number of samples to average over when making predictions.
    hyperparameters: dict; contains the hidden_units, l2 regularization factor and learning rate. If None, use default parameters that were used initially. check build_glmmnet_dependent or build_glmmnet_independent
    validation: if using validation split (for hyperparameter tuning) 
    """
    all_data = get_all_data(k, exp_id)
    val_scores = []
    for i in range(k):
        print(f"k-fold validation {i+1}/{k}")
        X_train, y_train, X_val, y_val = all_data[i]
        hicard_var = 'category'
        x_num = [col for col in X_train.columns if col not in hicard_var]
        colnames = [hicard_var] + x_num
        
        X_train = X_train.copy()
        X_val = X_val.copy()
        # Convert numeric to string so they can be recognised by ce.OrdinalEncoder
        X_train[hicard_var] = X_train[hicard_var].astype("str")
        X_val[hicard_var] = X_val[hicard_var].astype("str")

        ct_nn = make_column_transformer(
            (ce.ordinal.OrdinalEncoder(handle_unknown="value"), [hicard_var]), # "value" encodes unknown categories as "-1"
            (MinMaxScaler(feature_range=(0, 1)), x_num),
        )

        fix_missing_vales = make_column_transformer(
            (SimpleImputer(missing_values=-1, strategy="constant", fill_value=0), [hicard_var])
        )

        # Fit and transform the data using the ColumnTransformer
        X_train_ct = pd.DataFrame(ct_nn.fit_transform(X_train), columns= colnames)
        X_val_ct = pd.DataFrame(ct_nn.transform(X_val), columns= colnames)
        X_val_ct[hicard_var] = fix_missing_vales.fit_transform(X_val_ct)

        one_hot = ce.OneHotEncoder(cols=[hicard_var], handle_unknown="value")
        X_train_ct_ohe = one_hot.fit_transform(X_train_ct)
        X_val_ct_ohe = one_hot.transform(X_val_ct)
        
        X_embed_train, X_embed_val = embedding_preproc(X_train_ct, X_val_ct, [hicard_var])

        cardinality = len(X_train[hicard_var].unique()) 
        start = time.time()

        # Define training parameters
        epochs = 500
        patience = 50
        batch_size = 256
        es = EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss", verbose=2)

        NN_ee = build_baseline_nn_hyperparameters(X_train_ct, hyperparameters = hyperparameters, random_state=random_state, num_vars=x_num, cat_vars=[hicard_var])
        NN_ee.fit(
            (tuple(X_embed_train), X_train_ct.drop([hicard_var], axis=1)), y_train, validation_split=0.2, 
            epochs=epochs, callbacks=[es], batch_size=batch_size, verbose=0)

        y_pred_val = NN_ee.predict((tuple(X_embed_val), X_val_ct.drop([hicard_var], axis=1))).flatten()

        val_scores_i = evaluate_predictions(y_val, y_pred_val, categories=X_val[hicard_var], likelihood= likelihood)
        val_scores.append(val_scores_i)
        
    return {'val_scores': val_scores}

def NN_ee_hyperparameter_tuning_k_fold(exp_id, k, likelihood, n_search):
    tuning_results = []
    hyperparameters_tested = []
    val_scores_all = []
    
    learning_rate_range = 0.1**np.arange(1, 4, 1)  # Range for learning rate
    dropout_rate_range = [0.1, 0.2, 0.3]  # Range for dropout rate
    hidden_units_range = 2**np.arange(4,10,2)  # Range for number of units in hidden layers
    n_layers_range = np.arange(2,5)
    
    i = 0
    while i < n_search:
        if i == 0:
            hyperparameters = {'learning_rate': 0.01, 'dropout_rate': 0.2, 'n_layers': 3, 'hidden_units': [64, 32, 16]} # test default parameters
        else:
            # random search of hyperparameters
            learning_rate = random.choice(learning_rate_range) 
            dropout_rate = random.choice(dropout_rate_range)
            n_layers = random.choice(n_layers_range)
            hidden_units = random.choices(hidden_units_range, k = n_layers)
            hyperparameters = {'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'n_layers': n_layers, 'hidden_units': hidden_units}
        print(hyperparameters)
        if hyperparameters not in hyperparameters_tested:
            hyperparameters_tested.append(hyperparameters)
            print("yes")
            try:
                print(f"trying parameter search {i+1} / {n_search}")
                results = NN_ee_results_k_fold(exp_id, k, likelihood, hyperparameters)
                print("results")
                val_scores = pd.DataFrame(results["val_scores"])
                mean_val_scores = val_scores.mean()
                val_scores_all.append(val_scores)
                tuning_results.append(mean_val_scores)
                i += 1
            except:
                print("f")
                pass

            
    tuning_results = pd.DataFrame(tuning_results)
    top_hyperparameters = hyperparameters_tested[tuning_results['CRPS'].idxmin()]  
    return top_hyperparameters, tuning_results, hyperparameters_tested, val_scores_all

def get_all_data(k, exp_id): # get k-fold split data, stored as a list of length k. 
    Xy_train = pd.read_csv(f"data/experiment_{exp_id}/train_data.csv")
    Xy_test = pd.read_csv(f"data/experiment_{exp_id}/test_data.csv")

    data = pd.concat([Xy_train, Xy_test])
    X = data.drop(['y', 'y_true'], axis = 1)
    y = data['y']
    kf = KFold(n_splits=k, shuffle = True, random_state = 42)
    all_data = []

    for i, (train, val) in enumerate(kf.split(X)):
        X_train = X.iloc[train]
        y_train = y.iloc[train]

        X_val = X.iloc[val]
        y_val = y.iloc[val]

        all_data.append((X_train, y_train, X_val, y_val))
    return all_data

def get_train_test_scores_glmmnet(exp_id, regularizer, y_dist, top_hyperparameters, n_prediction_samples):
    train_scores_model = []
    test_scores_model = []

    for i in range(50):
        Xy_train = pd.read_csv(f"data/experiment_{exp_id}/train_data_{i + 1:02d}.csv")
        Xy_test = pd.read_csv(f"data/experiment_{exp_id}/test_data_{i + 1:02d}.csv")

        X_train = Xy_train.drop(['y', 'y_true'], axis = 1)
        y_train = Xy_train['y']

        X_test = Xy_test.drop(['y', 'y_true'], axis = 1)
        y_test = Xy_test['y']

        data = (X_train, y_train, X_test, y_test)

        runtimes, glmmnet, train_scores, test_scores = glmmnet_results(regularizer, n_prediction_samples, y_dist, data, top_hyperparameters)
        train_scores_model.append(train_scores)
        test_scores_model.append(test_scores)
        if i % 10 == 0:
            print(i)
    return train_scores_model, test_scores_model

def get_train_test_scores_nn_ee(exp_id, y_dist, top_hyperparameters):
    train_scores_model = []
    test_scores_model = []
    for i in range(50):
        Xy_train = pd.read_csv(f"data/experiment_{exp_id}/train_data_{i + 1:02d}.csv")
        Xy_test = pd.read_csv(f"data/experiment_{exp_id}/test_data_{i + 1:02d}.csv")

        X_train = Xy_train.drop(['y', 'y_true'], axis = 1)
        y_train = Xy_train['y']

        X_test = Xy_test.drop(['y', 'y_true'], axis = 1)
        y_test = Xy_test['y']

        data = (X_train, y_train, X_test, y_test)

        runtimes, extra_params, NN_ee, train_scores, test_scores = NN_ee_results(data, y_dist, top_hyperparameters)
        train_scores_model.append(train_scores)
        test_scores_model.append(train_scores)
        if i % 10 == 0:
            print(i)
            
    return train_scores_model, test_scores_model