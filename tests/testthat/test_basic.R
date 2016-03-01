library(deepboost)

context("basic functions")

data(adult, package='deepboost')

formula <- X..50K ~ X39 + X77516 + X13 + X2174 +  X0 + X40
levels(adult[,length(adult)]) <- c(1,-1)

train <- adult[1:29000,]
test <- adult[29001:32560,]

set.seed(666)

test_that("train and predict formula works", {
  bst <- deepboost.formula(formula, train, num_iter = 5)
  pred <- predict(bst, test)
  expect_equal(length(pred), 3560)
})

test_that("train and predict default works", {
  bst <- deepboost.default(train[,c("X39","X77516","X13")], train$X..50K, num_iter = 5)
  pred <- predict(bst, test)
  expect_equal(length(pred), 3560)
})

test_that("grid search works", {
  data("sonar")
  formula <- R ~ .
  best_params <-
    deepboost.gridSearch(formula, sonar, seed = 666, k = 2)

  expect_equal(best_params[1][[1]], 0.7634895)
  expect_equal(best_params[2][[1]], 5)
  expect_equal(best_params[3][[1]], 1e-04)
  expect_equal(best_params[4][[1]], 0.0625)
  expect_equal(best_params[5][[1]], "l")

})
