# Hand-on machine learning with R
# Chp 2. Modelling process
# July 2022

# Description: this script provide the R code to practice with the examples provided in the "Hands-on machine learning with R" book.
# The book can be found here: https://bradleyboehmke.github.io/HOML/intro.html#data

# |---- 0. Setup

# Helper packages
library(dplyr)     # for data manipulation
library(ggplot2)   # for awesome graphics

# Modeling process packages
library(rsample)   # for resampling procedures
library(caret)     # for resampling and model training
library(h2o)       # for resampling and model training

# h2o set-up 
h2o.no_progress()  # turn off h2o progress bars
h2o.init()         # launch h2o

# |---- 1. Data prep

# Ames housing data
ames <- AmesHousing::make_ames()
ames.h2o <- as.h2o(ames)

# Job attrition data
churn <- rsample::attrition %>% 
  mutate_if(is.ordered, .funs = factor, ordered = FALSE)
churn.h2o <- as.h2o(churn)