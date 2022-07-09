# ML with R
# Iris classification problem

# This script provides an exploration of classification methods and ML techniques used in R
# The use case starts from the IRIS classification problem as described here 
#    https://machinelearningmastery.com/machine-learning-in-r-step-by-step/

# 0. Set-up -------------

install.packages("caret")
install.packages("ellipse")

library(caret) # Upload pkg caret from ML methods
library(ellipse) # Pkg for ellipse representation in caret

# 1. Load dataset -------------

data(iris)
dataset <- iris

# 2. Create data partition -------------

validation_index <- caret::createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# 3. Dataset exploration -------------

# dimensions of dataset
dim(dataset)

# list types for each attribute
sapply(dataset, class)

# take a peek at the first 5 rows of the data
head(dataset)

# list the levels for the class
levels(dataset$Species)


# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# summarize attribute distributions
summary(dataset)

# 4. Visual exploration -------------

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]


# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

# barplot for class breakdown
plot(y, main = "Iris class breakdown")

# scatterplot matrix
caret::featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
caret::featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# 5. Classification model -------------

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Model building
# a) linear algorithms
set.seed(7)
fit.lda <- caret::train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- caret::train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- caret::train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- caret::train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- caret::train(Species~., data=dataset, method="rf", metric=metric, trControl=control)


# summarize accuracy of models
results <- caret::resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)
