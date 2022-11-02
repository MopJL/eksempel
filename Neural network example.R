install.packages("reticulate")
reticulate::install_miniconda()

install.packages("keras")
keras::install_keras()


library(keras)

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

max(sapply(train_data, max))


vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension) #Creates an all-zero matrix of shape (length(sequences), dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1 # Sets specific indices of results[i] to 1s
  results
}
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)


# convert labels to numeric

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# model definition

model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# configure the model

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


# set aside validation set

val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]


# training the model

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
str(history)
# this model is overfitting and we want to retrain a model from scratch to prevent this
# by only running it for 4 epochs instead of 20:

model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16 activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
results <- model %>% evaluate(x_test, y_test)
results

# loss 0.3 and accuracy 87%


head(x_train)
