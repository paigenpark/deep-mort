# installing and loading packages 
library(keras)
library(tensorflow)
library(whereami)
library(tidyverse)


### setting up DEEP6 (deep neural network with tanh activations)

# defining inputs
Year <- layer_input(shape = c(1), dtype = 'float32', name = 'Year')
Age <- layer_input(shape = c(1), dtype = 'int32', name = 'Age')
Country <- layer_input(shape = c(1), dtype = 'int32', name = 'Country')
Gender <- layer_input(shape = c(1), dtype = 'int32', name = 'Gender')

# defining embedding layers
# input data can take any integer value between 0 and 99
# each integer from the input will be mapped to vector of size 5
# each input sequence is a single integer

# summary: building layer that takes integer inputs representing age, gender,
# and country and embeds them into a dense space of 5 dimensions
# then flattens the output
Age_embed = Age %>%
  layer_embedding(input_dim = 100, output_dim = 5, input_length = 1, name = 'Age_embed') %>%
  keras::layer_flatten()

Gender_embed = Gender %>%
  layer_embedding(input_dim = 100, output_dim = 5, input_length = 1, name = 'Gender_embed') %>%
  keras::layer_flatten()

Country_embed = Country %>%
  layer_embedding(input_dim = 100, output_dim = 5, input_length = 1, name = 'Country_embed') %>%
  keras::layer_flatten()

# creating feature vector 
features <- layer_concatenate(list(Year, Age_embed, Gender_embed, Country_embed))

# setting up middle layers 
middle = features %>%
  layer_dense(units = 128, activation = 'tanh') %>%
  layer_batch_normalization() %>%
  layer_dropout(0.05) %>%
  
  layer_dense(units = 128, activation = 'tanh') %>%
  layer_batch_normalization() %>%
  layer_dropout(0.05) %>%
  
  layer_dense(units = 128, activation = 'tanh') %>%
  layer_batch_normalization() %>%
  layer_dropout(0.05) %>%
  
  layer_dense(units = 128, activation = 'tanh') %>%
  layer_batch_normalization() %>%
  layer_dropout(0.05)

# setting up output layer
main_output = layer_concatenate(list(features, middle)) %>%
  layer_dense(units = 128, activation = 'tanh') %>%
  layer_batch_normalization() %>%
  layer_dropout(0.05) %>%
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')

model <- keras_model(inputs = c(Year, Age, Country, Gender), outputs = c(main_output))

summary(model)


# compiling the model 
model %>% compile(loss = "mse",
                  optimizer = "adam",
                  metrics = c("accuracy"))

# fitting the model 
fit <- model %>%
  fit(x_train,
      y_train,
      epoch = 50,
      batch_size = 256,
      validation_split = 0.05,
      verbose = 2)
