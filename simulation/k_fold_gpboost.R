# k-fold validation for simulation data sets

# Read data
library(glue)
library(tidyverse)
library(gpboost)
library(brms)

y_dists = c("gaussian", "gamma", "gaussian", 
            "gaussian", "gaussian", "gamma")

objectives = c("regression", "gamma", "regression", 
               "regression", "regression", "gamma")

learning_rate_range = 0.1^c(2:5)
nrounds_range = 10^c(1:4)
num_leaves_range = 2^c(8:12)

min_data_in_leaf = 1000
max_depth = 1
# min_data_in_leaf_range = c(10, 100, 1000)
# max_depth_range = c(1, 2, 3, 5, 10)

set.seed(42)
n_samples <- 10
hyperparameters_tested_df <- data.frame(learning_rate = numeric(),
                                        nrounds = numeric(),
                                        num_leaves = numeric(),
                                        stringsAsFactors = FALSE)

while (dim(hyperparameters_tested_df)[1] < n_samples) {
  learning_rate <- sample(learning_rate_range, size = 1)
  nrounds <- sample(nrounds_range, size = 1)
  num_leaves <- sample(num_leaves_range, size = 1)
  
  hyperparameters <- data.frame(learning_rate = learning_rate, 
                                nrounds = nrounds,
                                num_leaves = num_leaves)
  
  # Check if hyperparameters already exist in the data frame
  if (!any(apply(hyperparameters_tested_df, 1, function(row) all(row == hyperparameters)))) {
    # Add hyperparameters to the data frame
    hyperparameters_tested_df <- rbind(hyperparameters_tested_df, hyperparameters)
  }
}

# getwd()
# write.csv(hyperparameters_tested_df, file = 'GPBoost_hyperparameters_tested.csv', row.names = F)
# hyperparameters_tested_df = read.csv("GPBoost_hyperparameters_tested.csv")

for (row in 1:dim(hyperparameters_tested_df)[1]){
  hyperparameters <- hyperparameters_tested_df[row, ]
  print(hyperparameters)
  print("exp_id")
  for (exp_id in 1:6){
    print(exp_id)
    y_dist = y_dists[exp_id]
    objective = objectives[exp_id]
    
    for (i in 1:4) {
      print(i)
      setwd(glue('data_ct/experiment_{exp_id}'))
      
      X_train_ct <- subset(read.csv(glue("k_fold_X_train_{i}.csv")), select = -X)
      X_val_ct <- subset(read.csv(glue("k_fold_X_val_{i}.csv")), select = -X)
      y_train <- read.csv(glue("k_fold_y_train_{i}.csv"))[,2]
      y_val <- read.csv(glue("k_fold_y_val_{i}.csv"))[,2]
      
      train_category = X_train_ct$category
      val_category = X_val_ct$category
      
      tryCatch({
        gp_model <- GPModel(group_data = train_category, likelihood = y_dist)
        gp_fit <- gpboost(
          data = as.matrix(X_train_ct), 
          label = y_train, 
          gp_model = gp_model,
          nrounds = hyperparameters$nrounds, 
          learning_rate = hyperparameters$learning_rate, 
          max_depth = max_depth,
          min_data_in_leaf = min_data_in_leaf,
          num_leaves = hyperparameters$num_leaves,
          use_nesterov_acc = TRUE, 
          verbose = 0,
          objective = objective)
        
        y_pred_val <- predict(
          gp_fit, data = as.matrix(X_val_ct), group_data_pred = val_category,
          predict_var = TRUE, pred_latent = FALSE, predict_cov_mat = FALSE
        )
        y_pred_val <- data.frame(y = y_pred_val$response_mean)
        
        setwd(glue('k_fold_results/experiment_{exp_id}/GPBoost'))
        write.csv(y_pred_val, file = glue("pred_val_k_fold_{row}_{i}.csv"), row.names = F)
        write.csv(gp_model$get_cov_pars()[[1]], file = glue("gpboost_scale_k_fold_{row}_{i}.csv"), row.names = F)
        write.csv(gp_model$get_aux_pars()[[1]], file = glue("gpboost_shape_k_fold_{row}_{i}.csv"), row.names = F)
      }, error = function(err) {
        cat("An error occurred:", conditionMessage(err), "\n")
      })
    }
  }
}