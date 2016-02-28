#' Returns optimised parameter list for deepboost model on given data
#'
#' @param object A Deepboost S4 class object
#' @param data input data.frame as training for model
#' @param pct_train percent of data used for training model (default 0.8)
#' @param seed for random split to train / test (default 666)
#' @details Finds optimised parameters for deepboost training.
#'  using grid search techniques over:
#'  - predefined, battle tested parameter possible values
#'  - cross validation over k folds
#' @return vectir with best accuracy value and a list of the best parameter combination (accuracy, (num_iter, beta, lambda, loss_type))
#' @export
deepboost.gridSearch <- function(object, data, pct_train = .8, seed=666) {

  if (!(is.numeric(pct_train)) || pct_train < 0.0 || pct_train > 1.0)
  {
    stop("ERROR_paramter_setting : beta must be between 0.0 and 1.0 and double (Default : 0.8)" )
  }

  num_iter_vals = c(5,10,25,50)
  beta_vals = c(2^-0, 2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6)
  lambda_vals = c(0.0001, 0.005, 0.01, 0.05, 0.1, 0.5)
  loss_type_vals = c("l","e")
  dpbGrid <-  expand.grid(num_iter = num_iter_vals,
                          beta = beta_vals,
                          lambda = lambda_vals,
                          loss_type = loss_type_vals)

  best_acc = -Inf

  set.seed(seed)
  train_ind <- sample(seq_len(nrow(data)), size = floor(pct_train * nrow(data)))

  eval_train <- data[train_ind, ]
  eval_test <- data[-train_ind, ]

  for(grow in 1:nrow(dpbGrid)){
    num_iter <- dpbGrid[grow,"num_iter"]
    beta <- dpbGrid[grow,"beta"]
    lambda <- dpbGrid[grow,"lambda"]
    loss_type <- dpbGrid[grow,"loss_type"]
    eval_model <- deepboost.formula(formula, eval_train, num_iter = num_iter, beta = beta, lambda = lambda, loss_type = loss_type)
    acc <-  sum(predict(eval_model, eval_test) == eval_test[,length(eval_test)]) / nrow(eval_test)
    if(acc > best_acc){
      best_acc <- acc
      best_num_iter <- num_ter
      best_lambda <- lambda
      best_beta <- beta
      best_loss_type <- loss_type
    }
  }

  RET <-
    c(best_acc,
      list(best_num_iter,
           best_lambda,
           best_beta,
           best_loss_type))

  return(RET)
}


# @param k number of folds (default = 10) for cross validation optimisation

#   if (!(is.numeric(k)) || k <= 1 || !(k%%1==0))
#   {
#     stop("ERROR_paramter_setting : k must be >= 2 and integer (Default : 10)" )
#   }

#   flds <- createFolds(1:nrow(ds), k)
#
#   for(fold in 1:k){
#     eval_train <- ds[-flds[[fold]],]
#     eval_test <- ds[flds[[fold]],]
