# ML with R
# Model Ensembling with CARET

# This script provides an exploration of Model Ensembling with CARET
# The use case is the Ionosphere dataset
# This script comes from this source here: 
#     https://machinelearningmastery.com/machine-learning-ensembles-with-r/

# . Notes

# Bagging -> Building multiple models (typically of the same type) from different subsamples of the training datase;
# Boosting -> Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the chain;
# Stacking -> Building multiple models (typically of differing types) and supervisor model that learns how to best combine the predictions of the primary models.

# 0. Set-up -------------

install.packages("caretEnsemble")

# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)

# Load the dataset
data(Ionosphere)
dataset <- Ionosphere
dataset <- dataset[,-2]
dataset$V1 <- as.numeric(as.character(dataset$V1))

head(dataset)

# 1. Boosting ---------------

# Example of Boosting Algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

# C5.0
set.seed(seed)
fit.c50 <- train(Class~., data=dataset, method="C5.0", metric=metric, trControl=control)

# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Class~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)

# summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)

# 2. Bagging ----------------

# Example of Bagging algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

# Bagged CART
set.seed(seed)
fit.treebag <- train(Class~., data=dataset, method="treebag", metric=metric, trControl=control)

# Random Forest
set.seed(seed)
fit.rf <- train(Class~., data=dataset, method="rf", metric=metric, trControl=control)

# summarize results
bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)

# 3. Stacking ---------------

# Example of Stacking algorithms
# create submodels
control <- caret::trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretEnsemble::caretList(Class~., data=dataset, trControl=control, methodList=algorithmList)
results <- caret::resamples(models)
summary(results)
dotplot(results)

# NOTE: when combining predictions of different models using stacking,
      # it is desirable for them to have low correlation, 
      # thus allowing for a new classifier to get the best from each model


# correlation between results
caret::modelCor(results)
splom(results)

# stack using glm
stackControl <- caret::trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretEnsemble::caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)
# Stacking with linear model

stack.rf <- caretEnsemble::caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)
# Stacking with linear model

