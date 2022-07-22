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

# 3.3 Dealing with missing values

AmesHousing::ames_raw %>% 
  filter(is.na(`Garage Type`)) %>% 
  select(`Garage Type`, `Garage Cars`, `Garage Area`)
# Garage Cars and Garage Area have 0s imputed when missing

# Visualize missing values
visdat::vis_miss(AmesHousing::ames_raw, cluster = TRUE)

# imputation based on median
ames_recipe %>%
  recipes::step_impute_median(Gr_Liv_Area)

# imputation based on kNN
ames_recipe %>%
  recipes::step_impute_knn(all_predictors(), neighbors = 6)

# imputated based on trees
ames_recipe %>%
  recipes::step_impute_bag(all_predictors())

# Checking for near-zero variance features (these features are most likely not informative from a rule-of-thumb approach)
caret::nearZeroVar(ames_train, saveMetrics = TRUE) %>% 
  tibble::rownames_to_column() %>% 
  filter(nzv)
# Rule of thumb
# 1. The fraction of unique values over the sample size is low (say ≤ 10%);
# 2- The ratio of the frequency of the most prevalent value to the frequency of the second most prevalent value is large (say ≥ 20%).

# 3.5 variables normalization
# Normalize all numeric columns (applies the YeoJohnson norm to all numeric variables)
recipe(Sale_Price ~ ., data = ames_train) %>%
  step_YeoJohnson(recipes::all_numeric())         

# normalization (recipe for both training and test on the same scale)
ames_recipe %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

# 3.6 Categorical feature engineering

# Lumping (group categoricals with few observations)
dplyr::count(ames_train, Neighborhood) %>% dplyr::arrange(n)
count(ames_train, Screen_Porch) %>% arrange(n) # disperse values

# Lump levels for two features (lumping into an "other" category)
lumping <- recipe(Sale_Price ~ ., data = ames_train) %>%
  recipes::step_other(Neighborhood, threshold = 0.01, 
             other = "other") %>%
  recipes::step_other(Screen_Porch, threshold = 0.1, 
             other = ">0")

# Apply this blue print --> you will learn about this at 
# the end of the chapter
apply_2_training <- recipes::prep(lumping, training = ames_train) %>%
  recipes::bake(ames_train)

# New distribution of Neighborhood (new and cleaned training data)
count(apply_2_training, Neighborhood) %>% arrange(n)

# New distribution of Screen_Porch
count(apply_2_training, Screen_Porch) %>% arrange(n)

