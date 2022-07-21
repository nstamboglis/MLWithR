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

# |---- 2. Target engineering

transformed_response <- log(ames_train$Sale_Price)
# Taake the log of the dependent variable to avoid 

# log transformation
ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_log(all_outcomes())
ames_recipe 
# Alternative approach: use a recipe to be applied later on

# Alternatives to taking logs (if dependent vaiable has negative records)
transformed_response_1p <- log1p(ames_train$Sale_Price)
# Adds 1 to the dependent variable prior taking logs

# Log transform a value
y <- log(10)

# Undo log-transformation 8used to translate the predicted results back to their original scale)
exp(y)
## [1] 10

# Box Cox transform a value
lambda <- 2
y <- forecast::BoxCox(10, lambda)

# Inverse Box Cox function
inv_box_cox <- function(x, lambda) {
  # for Box-Cox, lambda = 0 --> log transform
  if (lambda == 0) exp(x) else (lambda*x + 1)^(1/lambda) 
}

# Undo Box Cox-transformation (again to go back from prediction scale to original dependent variable scale)
inv_box_cox(y, lambda)
## [1] 10
## attr(,"lambda")
## [1] -0.03616899

# 3.3 misiing data
sum(is.na(AmesHousing::ames_raw))

# Missing values visualization
AmesHousing::ames_raw %>%
  is.na() %>%
  reshape2::melt() %>%
  ggplot(aes(Var2, Var1, fill=value)) + 
  geom_raster() + 
  coord_flip() +
  scale_y_continuous(NULL, expand = c(0, 0)) +
  scale_fill_grey(name = "", 
                  labels = c("Present", 
                             "Missing")) +
  xlab("Observation") +
  theme(axis.text.y  = element_text(size = 4))


