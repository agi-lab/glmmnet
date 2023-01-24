# R script to run GLMM (through `brms`) and GPBoost

# Update experiment parameters
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
exp_id <- 6                         # experiment id
y_dist <- "gamma"                   # distribution of the response variable y
overwrite <- TRUE

# Automatically update objective function
if (y_dist == "gaussian") {
  objective <- "regression"
} else if (y_dist == "gamma") {
  objective <- "gamma"
}

# Read data
library(glue)
library(tidyverse)
path <- glue("data/experiment_{exp_id}")
train_data <- read.csv(glue("{path}/train_data.csv"))
test_data <- read.csv(glue("{path}/test_data.csv"))

# Split data into X, y, y_true
split_data <- function(which_set) {
  df <- get(glue("{which_set}_data"))
  assign(glue("X_{which_set}"), select(df, -y, -y_true), envir = .GlobalEnv)
  # For compatibility, convert the y columns to matrices
  assign(glue("y_{which_set}"), as.matrix(select(df, y)), envir = .GlobalEnv)
  assign(glue("y_true_{which_set}"), as.matrix(select(df, y_true)), envir = .GlobalEnv)
}

for (set in c("train", "test")) {
  split_data(set)
}

# Run experiment
library(gpboost)
library(brms)

# Scale the input features
# Consistent with the Python scripts, use MinMax to scale numeric inputs to (0, 1)
MinMaxScaler <- function(x, x_train, a = 0, b = 1, ...) {
  a + (x - min(x_train, ...)) * (b - a) / (max(x_train, ...) - min(x_train, ...))
}
x_num <- colnames(select(X_train, -category))
X_test[x_num] <- 
  mapply(MinMaxScaler, X_test[x_num], X_train[x_num])
X_train[x_num] <- 
  mapply(MinMaxScaler, X_train[x_num], X_train[x_num])

# GLMM
start <- Sys.time()
glmm_fit <- brm(
  reformulate(c(x_num, "(1 | category)"), "y"),
  data = select(train_data, -y_true),
  family = y_dist,
  # iter = 4000, # uncomment to run longer if warning messages appear (exp #3)
  prior = prior(normal(0, 1), class = "b"),
  cores = parallel::detectCores(),
  seed = 42
)
end <- Sys.time()
brms_runtime <- end - start
s <- summary(glmm_fit)
re <- mixedup::extract_random_effects(glmm_fit)
re <- re[, "value"]

# Training set
y_pred_train <- predict(glmm_fit, cores = parallel::detectCores())
y_pred_train <- y_pred_train[, "Estimate"]

# Test set
y_pred_test <- predict(glmm_fit, newdata = test_data, cores = parallel::detectCores(), allow_new_levels = TRUE)
y_pred_test <- y_pred_test[, "Estimate"]

# Output predictions
if (overwrite) {
  write.csv(s$spec_pars$Estimate, file = glue("models/experiment_{exp_id}/brms_sigma.csv"), row.names = F)
  write.csv(re, file = glue("models/experiment_{exp_id}/brms_re.csv"), row.names = F)
  write.csv(y_pred_train, file = glue("models/experiment_{exp_id}/brms_pred_train.csv"), row.names = F)
  write.csv(y_pred_test, file = glue("models/experiment_{exp_id}/brms_pred_test.csv"), row.names = F)
}

# Define model structure: random intercept model
set.seed(42)
gp_model <- GPModel(group_data = X_train$category, likelihood = y_dist)

# We'll skip systematic hyperparameter tuning and go straight to training
# The hyperparameters below are selected based on a few trial runs
start <- Sys.time()
gp_fit <- gpboost(
  data = as.matrix(X_train), label = y_train, gp_model = gp_model,
  nrounds = 100, learning_rate = 0.01, max_depth = 1,
  use_nesterov_acc = TRUE, verbose = 0,
  objective = objective
)
end <- Sys.time()
gpb_runtime <- end - start

# Check covariance parameters
# Estimated covariance parameters, i.e. sigma and sigma_1
summary(gp_model)

# Output model predictions on both the training and the test sets
y_pred_train <- predict(
  gp_fit, data = as.matrix(X_train), group_data_pred = X_train$category,
  predict_var = TRUE, pred_latent = FALSE, predict_cov_mat = FALSE
)
# From Documentation (https://gpboost.readthedocs.io/): 
# result["response_mean"] are the predicted means of the response variable 
# (label) taking into account both the fixed effects (tree-ensemble) and the 
# random effects (gp_model)
y_pred_train <- data.frame(y = y_pred_train$response_mean)

# Test set
y_pred_test <- predict(
  gp_fit, data = as.matrix(X_test), group_data_pred = X_test$category,
  predict_var = TRUE, pred_latent = FALSE, predict_cov_mat = FALSE
)
y_pred_test <- data.frame(y = y_pred_test$response_mean)

# Output random effect predictions
re <- predict(
  gp_model, group_data_pred = X_train$category, 
  X_pred = as.matrix(X_train),
  predict_var = TRUE, predict_cov_mat = FALSE)
re <- re$mu # Predicted mean

# Output predictions
if (overwrite) {
  write.csv(gp_model$get_cov_pars()[[1]], file = glue("models/experiment_{exp_id}/gpboost_scale.csv"), row.names = F)
  write.csv(re, file = glue("models/experiment_{exp_id}/gpboost_re.csv"), row.names = F)
  write.csv(y_pred_train, file = glue("models/experiment_{exp_id}/gpboost_pred_train.csv"), row.names = F)
  write.csv(y_pred_test, file = glue("models/experiment_{exp_id}/gpboost_pred_test.csv"), row.names = F)
}

