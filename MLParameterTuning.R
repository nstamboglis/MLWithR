# ML with R
# Hyper-parameter tuning

# This script provides an exploration of Model hyper-parameter tuning with CARET
# The use case is the Sonar dataset
# This script comes from this source here: 
#     https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/

# . Notes
# - The script tunes a random forest model. The tuned hyper parameters are:

# 1. mtry: Number of variables randomly sampled as candidates at each split;
# 2. ntree: Number of trees to grow.

# 0. Set-up -------------

library(randomForest)
library(mlbench)
library(caret)

# Load Dataset
data(Sonar)
dataset <- Sonar
x <- dataset[,1:60]
y <- dataset[,61]

head(dataset)
table(dataset$Class)

# 1. Tune baseline parameter |----------------

# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- caret::train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

# 2. |--- Random search tuning |----
# Random Search
control <- caret::trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol(x))
rf_random <- caret::train(Class~., data=dataset, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# 2. |--- Grid search tuning |----
# Grid Search. 
# NOTE: In this case the hyper parameter space is a line as we're searching for mtry only
control <- caret::trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15)) # hyper parameter seach grid
rf_gridsearch <- caret::train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

# 3. |--- Tuning using randomForest package |----

# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry <- randomForest::tuneRF(x, y, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)

# 4. |--- Manual tuning |----

# Manual Search
control <- caret::trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(x))))
modellist <- list()
for (ntree in c(1000, 1500, 2000, 2500)) {
  set.seed(seed)
  fit <- caret::train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)

# 5. |--- Extend Caret |----

# Extend caret's RF algo by allowing for the support of multiple tuning of multiple parameters

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# train model
control <- trainControl(method="repeatedcv", number=10, repeats=3)
tunegrid <- expand.grid(.mtry=c(1:15), .ntree=c(1000, 1500, 2000, 2500))
set.seed(seed)
custom <- train(Class~., data=dataset, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
summary(custom)
plot(custom)

