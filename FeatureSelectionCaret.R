# ML with R
# Feature selection with CARET

# This script provides an exploration of Feature Selection techniques with CARET
# The use case is the Pima Indians Diabetes dataset
# This script comes from this source here: 
#     https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

# 0. Set-up -------------

install.packages("caret") # install ML pkg
install.packages("corrplot") # Install pkg for correlation plots
install.packages("glmnet") # Install pkg for Logit regression
install.packages("mlbench") # Install pkg for ML real-life use cases
install.packages("tidyverse") # No presentation needed

library(caret) # Upload pkg caret from ML methods
library(corrplot) # Upload pkg for correlation plots
library(glmnet) # Upload pkg for Logit regression
library(mlbench) # Pkg for real-life ML example
library(tidyverse) # No presentation needed

# 1. Exploratory analysis -------------

# ensure the results are repeatable
set.seed(7)
# load the data
data(PimaIndiansDiabetes)

# Explore dataset
str(PimaIndiansDiabetes)
summary(PimaIndiansDiabetes)

table(PimaIndiansDiabetes$diabetes) # Positive lower than negative -> unbalanced dataset

# boxplot for each attribute on one image
par(mfrow=c(1,8))
for(i in 1:8) {
  boxplot(PimaIndiansDiabetes[,i], main=names(PimaIndiansDiabetes)[i])
}
# Different scales and variability in the data. Perhaps a scaling is needed?

x <- PimaIndiansDiabetes[,1:8]# data frame of dependent variables
y <- PimaIndiansDiabetes[,9] # data frame of independent variable

# scatterplot matrix
caret::featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
caret::featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# Several features seem to overalp across the two groups. Not all features appear to be discriminant in predicting diabetes.

# 2. Correlation analysis -------------

# calculate correlation matrix
correlationMatrix <- cor(PimaIndiansDiabetes[,1:8])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- caret::findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(colnames(PimaIndiansDiabetes[,1:8])[highlyCorrelated])
# Age appears to be correlated with at least one other variable (corr higher than cutoff of 0.5)

corrplot(correlationMatrix, method = 'number') # colorful number
# Visual exploration of the corr diagram
# From here we clearly see that "age" is higly correlated with "pregnant"

# 3. Feature importance -------------

# prepare training scheme
control <- caret::trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- caret::varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# 4. Feature importance using a Recursive Feature Elimination -------------

# define the control using a random forest selection function
controlRfe <- caret::rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- caret::rfe(PimaIndiansDiabetes[,1:8], PimaIndiansDiabetes$diabetes, sizes=c(1:8), rfeControl=controlRfe)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

# Notice: among the most important features we have some highly correlated ones, such as "age" and "pregnant"..

# 5. Extra: select best fitting model and make predictions -------------

# Create a validation dataset
validation_index <- caret::createDataPartition(PimaIndiansDiabetes$diabetes, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- PimaIndiansDiabetes[-validation_index,]
# use the remaining 80% of data to training and testing the models
PimaIndiansDiabetesTraining <- PimaIndiansDiabetes[validation_index,]

# Run algorithms using 10-fold cross validation
#controlFit <- caret::trainControl(method="cv", number=10)
controlFit <- caret::trainControl(method="repeatedcv", number=10, repeats = 3)
metric <- "Accuracy"

# Model building

# Learning Vector Quantization (LVQ)
fit.lvq <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="lvq", preProcess="scale", trControl=control)

# a) linear algorithms
set.seed(7)
fit.lda <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="lda", trControl=controlFit, preProcess = "scale")
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="rpart", metric=metric, trControl=controlFit, preProcess = "scale")
# kNN
set.seed(7)
fit.knn <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="knn", metric=metric, trControl=controlFit, preProcess = "scale")
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="svmRadial", metric=metric, trControl=controlFit, preProcess = "scale")
# Random Forest
set.seed(7)
fit.rf <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="rf", metric=metric, trControl=controlFit, preProcess = "scale")

# Logit model
set.seed(7)
fit.logit <- caret::train(diabetes~., data=PimaIndiansDiabetes, method="glmnet", metric=metric, trControl=controlFit, preProcess = "scale", family = 'binomial')

# summarize accuracy of models
results <- caret::resamples(list(lvq = fit.lvq, lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf, logit = fit.logit))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.logit)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.logit, validation)
confusionMatrix(predictions, validation$diabetes)

# 6. Extras: select best performing model (logit) without correlated variable age
fit.logitNoAge <- caret::train(diabetes~., data=PimaIndiansDiabetes|>select(-age) , method="glmnet", metric=metric, trControl=controlFit, preProcess = "scale", family = 'binomial')
fit.logitNoPregnant <- caret::train(diabetes~., data=PimaIndiansDiabetes|>select(-pregnant) , method="glmnet", metric=metric, trControl=controlFit, preProcess = "scale", family = 'binomial')

# summarize accuracy of Logit models
results2 <- caret::resamples(list(logit = fit.logit, logitNoAge = fit.logitNoAge, logitNoPregnant = fit.logitNoPregnant))
summary(results2)

# compare accuracy of models
dotplot(results2)

# summarize Best Model: complete logit model still the best one
print(fit.logit)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.logit, validation)
confusionMatrix(predictions, validation$diabetes)

# Printing out other confusion matrices for curiosity
predictionsNoAge <- predict(fit.logitNoAge, validation)
confusionMatrix(predictionsNoAge, validation$diabetes)

predictionsNopregnant <- predict(fit.logitNoPregnant, validation)
confusionMatrix(predictionsNopregnant, validation$diabetes)
