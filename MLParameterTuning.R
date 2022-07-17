# ML with R
# Hyper-parameter tuning

# This script provides an exploration of Model hyper-parameter tuning with CARET
# The use case is the Sonar dataset
# This script comes from this source here: 
#     https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/

# . Notes


# 0. Set-up -------------

install.packages("caretEnsemble")

# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)

# Load the dataset