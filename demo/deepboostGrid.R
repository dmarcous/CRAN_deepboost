data("sonar")
formula <- R ~ .
best_params <-
  deepboost.gridSearch(formula, sonar)
