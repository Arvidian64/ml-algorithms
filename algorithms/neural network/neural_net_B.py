import numpy as np
import math
import keras
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

def load_data():
    data = np.loadtxt('maintenance.txt')
    X_features = data[:, :16]      #16 features
    y_coefficients = data[:, 16:]  #2 decay state coefficients
    return X_features, y_coefficients

def prep_data(rng, X_features, y_coefficients):
    n = len(X_features)       
    perm = np.random.permutation(n) 
    X_features = X_features[perm] #ADDED so features and coefficients are randomly shuffled in the same order to "randomly be divided into training set..."
    y_coefficients = y_coefficients[perm]
    #splits data into 50% training, 25% validation, 25% testing
    split_50 = int(n * 0.5)
    split_25 = int(n * 0.75)

    X_train = X_features[:split_50]
    y_train = y_coefficients[:split_50]

    X_valid = X_features [split_50:split_25]
    y_valid = y_coefficients[split_50:split_25]

    X_test = X_features[split_25:]
    y_test = y_coefficients[split_25:]

    #Standardize & avoid 0-division
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std) 

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std = np.where(y_std == 0, 1.0, y_std)

    X_train = (X_train - mean) /std
    X_valid = (X_valid - mean) /std 
    X_test = (X_test - mean) /std 

    #standardizing output values gave better results
    y_train = (y_train - y_mean) /y_std
    y_valid = (y_valid - y_mean) /y_std

    return X_train, X_valid, X_test, y_train, y_valid, y_test, y_mean, y_std

def model_architecture():
    input_layer = Input(shape=(16, ))
    dense = Dense(256, activation='relu')(input_layer)
    dense_1 = Dense(256, activation='relu')(dense)
    dense_2 = Dense(2, activation='linear')(dense_1)
    model = Model(inputs=input_layer, outputs=dense_2)
    return model

def get_weights(model):
    w1 = model.get_layer('dense').get_weights()[0]
    b1 = model.get_layer('dense').get_weights()[1]

    w2 = model.get_layer('dense_1').get_weights()[0]
    b2 = model.get_layer('dense_1').get_weights()[1] 

    w3 = model.get_layer('dense_2').get_weights()[0]
    b3 = model.get_layer('dense_2').get_weights()[1]

    return (w1, b1), (w2, b2), (w3, b3)

def forward(weights, X):
    (w1, b1), (w2, b2), (w3, b3) = weights
    hidden_1 = np.maximum(0, np.dot(X, w1) + b1)
    hidden_2 = np.maximum(0, np.dot(hidden_1, w2) + b2)
    output = np.dot(hidden_2, w3) + b3
    values = (X, hidden_1, hidden_2)
    return output, values

def mean_squared_error(output, target_values):
    return np.mean((output - target_values) ** 2)

def backpropagation(weights, X_train, y_train, learning_rate):
    (w1, b1), (w2, b2), (w3, b3) = weights 
    predictions, (inputs, hidden_1, hidden_2) = forward(weights, X_train)

    output_gradient = 2 * (predictions - y_train) / predictions.size

    w3_gradient = np.dot(hidden_2.T, output_gradient)
    b3_gradient = output_gradient.sum(axis=0)

    hidden_2_gradient = np.dot(output_gradient, w3.T)
    hidden_2_gradient *= hidden_2 > 0

    w2_gradient = np.dot(hidden_1.T, hidden_2_gradient)
    b2_gradient = hidden_2_gradient.sum(axis=0)

    hidden_1_gradient = np.dot(hidden_2_gradient, w2.T)
    hidden_1_gradient *= hidden_1 > 0

    w1_gradient = np.dot(inputs.T, hidden_1_gradient)
    b1_gradient = hidden_1_gradient.sum(axis=0)

    updated_weights = (
        (w1 - learning_rate * w1_gradient, b1 - learning_rate * b1_gradient),
        (w2 - learning_rate * w2_gradient, b2 - learning_rate * b2_gradient),
        (w3 - learning_rate * w3_gradient, b3 - learning_rate * b3_gradient),
    )
    return updated_weights


def run_model(rng, X_train, y_train, X_valid, y_valid, X_test, y_test, y_mean, y_std, weights):
    epochs = 5000
    batch_size = 16
    learning_rate = 0.004
    epoch_training_loss = [] 
    epoch_validation_loss = [] #for a figure?

    for epoch in range(epochs):
       #if epoch == 50:
            #learning_rate = learning_rate / 2
        print('Epoch:', epoch)

        ##shuffle X and make batches from X and y
        permutation = list(np.random.permutation(len(X_train)))
        shuffled_X_train = X_train[permutation, :]
        shuffled_y_train = y_train[permutation, :]
    
        num_mini_batches = math.ceil(len(X_train) / batch_size)
    
        for i in range(num_mini_batches):
            start = batch_size * i
            end = batch_size * (i + 1)
            X_batch_train = shuffled_X_train[start:end, :] 
            y_batch_train = shuffled_y_train[start:end, :]
    
            #Make predictions, calculate how off they are - replaces optimizer + gradient tape
            weights = backpropagation(weights, X_batch_train, y_batch_train, learning_rate)

        #Calculate how weights should change + update weights
        train_predictions = forward(weights, X_train)[0]
        validation_predictions = forward(weights, X_valid)[0]

        training_loss = mean_squared_error(train_predictions, y_train)
        epoch_training_loss.append(training_loss)
        validation_loss = mean_squared_error(validation_predictions, y_valid)
        epoch_validation_loss.append(validation_loss)
        print(f'Training loss over epoch: {float(training_loss):.9f}')
        print(f'Validation loss over epoch: {float(validation_loss):.9f}')
        print('------- -- End of epochs -- -------')

    plots(epoch_training_loss, epoch_validation_loss)

    ##running model on test data & scale back
    predict_scaled = forward(weights, X_test)[0]
    predict = predict_scaled * y_std + y_mean

    mean_sq_error_compressor = np.mean((y_test[:, 0] - predict[:, 0]) ** 2)
    mean_sq_error_turbine = np.mean((y_test[:, 1] - predict[:, 1]) ** 2)

    if mean_sq_error_compressor <= (5 * (10**(-8))):
        print(f'Decay compressor coefficient passed! Value: {mean_sq_error_compressor:.9f}')
    else:
        print(f'Decay compressor coefficient did not pass... Value: {mean_sq_error_compressor:.12e}')

    if mean_sq_error_turbine <= (1.5 * (10**(-8))):
        print(f'Decay turbine coefficient passed! Value: {mean_sq_error_turbine:.9f}')
    else:
        print(f'Decay turbine coefficient did not pass... Value: {mean_sq_error_turbine:.12e}')

def plots(epoch_training_loss, epoch_validation_loss):
    plt.figure(figsize=(15, 10))
    plt.plot(epoch_training_loss, label='Training loss')
    plt.plot(epoch_validation_loss, label='Validation loss')
    plt.title('Training and validation loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plot.png')

def main():
    X_features, y_coefficients = load_data()
    rng = np.random.default_rng(42)
    X_train, X_valid, X_test, y_train, y_valid, y_test, y_mean, y_std = prep_data(rng, X_features, y_coefficients)
    keras.utils.set_random_seed(42)
    model = model_architecture()
    weights = get_weights(model)
    #forward(weights, X_train)
    #backpropagation(weights, X_train, y_train, learning_rate=0.001)
    run_model(rng, X_train, y_train, X_valid, y_valid, X_test, y_test, y_mean, y_std, weights)
    
if __name__ == "__main__":
    main()
