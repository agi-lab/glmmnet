import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
from scipy.linalg import block_diag
from tqdm import tqdm

tfd = tfp.distributions

# Create a non-negative weight constraint instance
# Ref: https://www.tensorflow.org/api_docs/python/tf/keras/constraints/Constraint
class NonNegative(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

# Create a layer that computes the distributional parameters:
#    - location parameter (mu): g^-1(f(X) + RE), where g^-1 is specified by the inverse_link argument
#    - scale parameter (phi > 0), shared across all observations (e.g. sigma^2 in the Gaussian case)
# To be used after f(X) and RE are computed and added together
class DistParams(layers.Layer):
    def __init__(self, inverse_link="identity", phi_init=None, **kwargs) -> None:
        """
        Parameters
        ----------
        inverse_link: str
            Inverse link function. Default is "identity".
            One of ["exp", "identity", "log", "inverse"].
        phi_init: float or tf.keras.initializers.Initializer
            Initial value of phi (scale parameter).
        **kwargs: 
            Keyword arguments for the parent class, e.g. name.
        """
        super(DistParams, self).__init__(**kwargs)

        # Validate inputs
        inverse_link_options = ["exp", "identity", "log", "inverse"]
        if inverse_link not in inverse_link_options:
            raise ValueError(f"Expected {inverse_link!r} to be one of {inverse_link_options!r}")
        
        # Define layer attributes
        self.inverse_link = inverse_link
        if self.inverse_link == "exp":
            self.inverse_link_fn = tf.math.exp
        elif self.inverse_link == "identity":
            self.inverse_link_fn = tf.identity
        elif self.inverse_link == "log":
            self.inverse_link_fn = tf.math.log
        elif self.inverse_link == "inverse":
            self.inverse_link_fn = tf.math.reciprocal
        
        # Create layer weights that do not depend on input shapes
        # Add phi (dispersion parameter)
        self.phi_init = phi_init
        if isinstance(phi_init, float) or isinstance(phi_init, int):
            phi_init = tf.constant_initializer(phi_init)
        self.phi = self.add_weight(
            name="phi",
            shape=(1,),
            initializer=phi_init if phi_init is not None else "ones",
            constraint=NonNegative(),
            trainable=True,
        )

    def call(self, eta):
        """
        Apply the inverse link function to f(X) + Zu, to obtain the conditional mean of y|u.
        Input: eta = f(X) + Zu.
        Output: loc_param = g^-1(eta), scale_param.
        """
        loc_param = self.inverse_link_fn(eta)
        scale_param = tf.expand_dims(tf.repeat(self.phi, tf.shape(eta)[0]), axis=1)
        return tf.concat([loc_param, scale_param], axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "inverse_link": self.inverse_link,
            "phi_init": self.phi_init,
        })
        return config

def build_glmmnet(cardinality, num_vars, final_layer_likelihood, train_size, random_state=42, regularizer=False, is_prior_trainable=False):
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

    # Construct a standard FFNN for the fixed effects
    num_inputs = layers.Input(shape=(len(num_vars),), name="numeric_inputs")
    hidden_units = [64, 32, 16]
    hidden_activation = "relu"
    x = num_inputs
    for hidden_layer in range(len(hidden_units)):
        units = hidden_units[hidden_layer]
        x = layers.Dense(
            units=units, activation=hidden_activation, 
            kernel_regularizer=tf.keras.regularizers.l2(0.01) if regularizer else None,
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
        activity_regularizer=tf.keras.regularizers.l2(0.01) if regularizer else None,
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
    glmmnet.compile(optimizer="adam", loss=NLL)

    return glmmnet

def predict_glmmnet(glmmnet, data, hicard_var, n_prediction_samples = 100):
    """
    Predict the response variable for a given dataset from a fitted glmmnet model.
    To make predictions, we call the model multiple times and average the results.
    The randomness comes from that the RE are sampled from the posterior (in the DenseVariational layer).

    Parameters
    ----------
    glmmnet: a fitted glmmnet model.
    data: a pandas dataframe with the same columns as the training data.
    hicard_var: the name of the high-cardinality variable.
    n_prediction_samples: the number of samples to average over when making predictions.
    """
    y_pred = np.zeros((data.shape[0], n_prediction_samples))
    hicard_columns = data.columns[data.columns.str.startswith(hicard_var)]
    for i in tqdm(range(n_prediction_samples)):
        y_pred[:, i] = glmmnet((
            tf.convert_to_tensor(data.drop(hicard_columns, axis=1)), 
            tf.convert_to_tensor(data[hicard_columns])), training=False).mean().numpy().flatten()
    y_pred = y_pred.mean(axis=1)

    return y_pred

def build_baseline_nn(X_train, objective="mse", print_embeddings=False, random_state=42, cat_vars=[], num_vars=[]):
    """
    Build a baseline neural network model with embeddings.
    Code adapted from https://github.com/oegedijk/keras-embeddings/blob/72c1cfa29b1c57b5a14c24781f9dc713becb68ec/build_embeddings.py#L38
    """
    tf.random.set_seed(random_state)
    inputs = []
    embeddings = []

    for col in cat_vars:
        # Estimate cardinality on the training set
        cardinality = int(np.ceil(X_train[col].nunique()))
        # Set the embedding dimension
        embedding_dim = int(max(cardinality ** (1/4), 2))
        if print_embeddings:
            print(f'[{col}] cardinality: {cardinality} and embedding dim: {embedding_dim}')
        
        # Construct the embedding layer
        col_inputs = layers.Input(shape=(1, ), name=col+"_input")
        embedding = layers.Embedding(input_dim=cardinality, output_dim=embedding_dim, input_length=1, name=col+"_embed")(col_inputs)
        # Use SpatialDropout to prevent overfitting
        # See: https://stackoverflow.com/questions/50393666/how-to-understand-spatialdropout1d-and-when-to-use-it
        embedding = layers.SpatialDropout1D(0.2, name=col+"dropout")(embedding)
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
    hidden_units = [64, 32, 16]
    hidden_activation = "relu"
    output_activation = "linear"
    for hidden_layer in range(len(hidden_units)):
        units = hidden_units[hidden_layer]
        x = layers.Dense(units=units, activation=hidden_activation, name=f"hidden_{hidden_layer + 1}")(x)
    output = layers.Dense(units=1, activation=output_activation, name="output")(x)
    model = models.Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer="adam", loss=objective, metrics=["mae"])

    return model

def _get_block_cov_matrix(block_sizes, cov_pars):
    """
    Creates lower-triangular matrix for block diagonal covariance matrix. covariance = scale_tril * scale_tril^T

    Parameters
    ----------
    block_sizes: list of sizes of adjacent dependent blocks
    cov_pars: tensor containing inputs for lower triangular matrix for block diagonal covariance matrix
    """
    tril_indices = None  # Initialize tril_indices

    start = 0
    offset = [0, 0]
    for i in range(len(block_sizes)):
        size = block_sizes[i]
        end = start + size * (size + 1) // 2
        indices = tf.where(tf.linalg.band_part(tf.ones((size, size)), -1, 0)) + offset
        if tril_indices is None:
            tril_indices = indices
        else:
            tril_indices = tf.concat([tril_indices, indices], axis=0)
        offset = [size + j for j in offset]
        start = end
    # reshaped_tensors = [tf.reshape(t, (-1, 2)) for t in tril_indices]
    # concatenated = tf.concat(reshaped_tensors, axis=0)
        
    # The size of tril_indices is sum of n * (n + 1) / 2 for n in block_sizes
    # Check if the size of tril_indices is equal to the size of cov_pars
    assert len(tril_indices) == len(cov_pars), "The size of cov_pars does not match up with the size of the lower triangular matrix specified."

    block_matrix = tf.scatter_nd(
        indices=tril_indices,
        updates=cov_pars,
        shape=(sum(block_sizes), sum(block_sizes))
    )
    return block_matrix

def build_glmmnet_dependent(cardinality, num_vars, final_layer_likelihood, train_size, block_sizes, random_state=42, regularizer=False, is_prior_trainable=False):
    """
    Build a GLMMNet model allowing for posterior correlation.

    Parameters
    ----------
    cardinality: int, number of unique values in the high-cardinality variable of interest
    num_vars: list of numerical variables
    final_layer_likelihood: str, likelihood of the final layer, "gaussian" or "gamma"
    train_size: int, number of training samples (used in kl_weight)
    block_sizes: list of sizes of adjacent dependent blocks
    random_state: int, random seed
    regularizer: bool, whether to use l2 regulariser, default is False
    is_prior_trainable: bool, whether to train the prior parameters, default is False
    """
    # Set random seed
    tf.random.set_seed(random_state)

    # Construct a standard FFNN for the fixed effects
    num_inputs = layers.Input(shape=(len(num_vars),), name="numeric_inputs")
    hidden_units = [64, 32, 16]
    hidden_activation = "relu"
    x = num_inputs
    for hidden_layer in range(len(hidden_units)):
        units = hidden_units[hidden_layer]
        x = layers.Dense(
            units=units, activation=hidden_activation, 
            kernel_regularizer=tf.keras.regularizers.l2(0.01) if regularizer else None,
            name=f"hidden_{hidden_layer + 1}")(x)
    f_X = layers.Dense(units=1, activation="linear", name="f_X")(x)

    # Deal with categorical inputs (random effects)
    cat_inputs = layers.Input(shape=(cardinality,), name=f"category_OHE_inputs")
    
    # Specify the surrogate posterior over the random effects
    def posterior_u(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size # Size of categorical variable
        num_cov_pars = int(sum([size*(size+1)/2 for size in block_sizes]))

        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n + num_cov_pars, dtype=dtype, initializer="random_normal"), #[:n] outputs get passed to loc parameters. [n:] outputs get passed as cov parameters 
            tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalTriL(
                    loc = t[..., :n], 
                    scale_tril = 0.01 * _get_block_cov_matrix(block_sizes, t[...,n:]) #tf.convert_to_tensor(_get_block_cov_matrix(block_sizes, t[...,n:]), dtype = dtype)
            ))
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
    # Construct the random effects, by variational inference
    RE = tfp.layers.DenseVariational(
        units=1, 
        make_posterior_fn=posterior_u,
        make_prior_fn=prior_trainable if is_prior_trainable else prior_fixed,
        kl_weight=1 / train_size,
        use_bias=False,
        activity_regularizer=tf.keras.regularizers.l2(0.01) if regularizer else None,
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
    glmmnet.compile(optimizer="adam", loss=NLL)

    return glmmnet