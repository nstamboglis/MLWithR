# Hand-on machine learning with R
# Chp 1. Introduction
# July 2022

# Description: this script provide the R code to practice with the examples provided in the "Hands-on machine learning with R" book.
# The book can be found here: https://bradleyboehmke.github.io/HOML/intro.html#data

# |---- 0. Setup

install.packages("AmesHousing")
install.packages("dslabs")
install.packages("modeldata")
install.packages("rsample")

library(AmesHousing)
library(dslabs)
library(modeldata)
library(rsample)
library(tidyverse)

# |---- 0. datasets


# |---- AMES HOUSING

# access data
ames <- AmesHousing::make_ames()
# initial dimension
dim(ames)
## [1] 2930   81
# response variable
head(ames)
## [1] 215000 105000 172000 244000 189900 195500

# |---- IBM Employee Attrition

# access data
attrition <- modeldata::attrition
# NOTE: Attrition data now in modeldata package

# initial dimension
dim(attrition)
## [1] 1470   31

# response variable
head(attrition$Attrition)
table(attrition$Attrition)
View(attrition)

# |---- MNIST for image recognition

#access data
mnist <- dslabs::read_mnist()
names(mnist)
## [1] "train" "test"

# initial feature dimensions
dim(mnist$train$images)
## [1] 60000   784

# response variable
head(mnist$train$labels)
## [1] 5 0 4 1 9 2

# |---- GROCERY data for market basket analysis

# URL to download/read in the data
url <- "https://koalaverse.github.io/homlr/data/my_basket.csv"

# Access data
my_basket <- readr::read_csv(url)

# Print dimensions
dim(my_basket)
## [1] 2000   42

# Peek at response variable
my_basket
