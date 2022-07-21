# Hand-on machine learning with R
# Chp 3. Feature engineering
# July 2022

# Description: this script provide the R code to practice with the examples provided in the "Hands-on machine learning with R" book.
# The book can be found here: https://bradleyboehmke.github.io/HOML/intro.html#data

# Foundation script checking the modelling data approach
# Dataset used:
# - Ames housing for regression
# - IBM attrition for classification

# |---- 0. Setup

# Helper packages
library(dplyr)     # for data manipulation
library(ggplot2)   # for awesome graphics
library(visdat)   # for additional visualizations

# Feature engineering packages
library(caret)    # for various ML tasks
library(recipes)  # for feature engineering tasks

# h2o set-up 
h2o.no_progress()  # turn off h2o progress bars
h2o.init()         # launch h2o

# |---- 1. Data prep

# Ames housing data
ames <- AmesHousing::make_ames()
ames.h2o <- as.h2o(ames)

# Stratified sampling with the rsample package
set.seed(123)
split <- rsample::initial_split(ames, prop = 0.7, 
                                strata = "Sale_Price")
ames_train  <- rsample::training(split)
ames_test   <- rsample::testing(split)