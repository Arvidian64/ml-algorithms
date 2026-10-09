import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model

def load_data():
    data = np.loadtxt('maintenance.dat')
    X_features = data[:, :16]      #16 features
    y_coefficients = data[:, 16:]  #2 decay state coefficients
    return X_features, y_coefficients

def prep_data(X_features, y_coefficients):
    n = len(X_features)
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

    X_train = (X_train - mean) /std
    X_valid = (X_valid - mean) /std 
    X_test = (X_test - mean) /std 

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def model_architecture():
    input = Input(shape=(16, ))
    hidden_1 = Dense(64, activation='relu')(input)
    hidden_2 = Dense(64, activation='relu')(hidden_1)
    output = Dense(2, activation='linear')(hidden_2)
    model = Model(inputs=input, outputs=output)

    return model

def run_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    ##training##
    epochs = 3
    #training_loss= [] 
    #validation_loss = [] #for a figure?

    #Those should maybe be written by ourselves? 
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(32)

    #metrics - should be written by ourselves?
    train_metric = keras.metrics.MeanSquaredError()
    valid_metric = keras.metrics.MeanSquaredError()

    for epoch in range(epochs):
        print('Epoch:', epoch)

        for step, (X_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                #Make predictions, calculate how off they are
                predictions = model(X_batch_train, training=True)
                loss_value = loss_function(y_batch_train, predictions)

            #Calculate how weights should change + update weights
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            train_metric.update_state(y_batch_train, predictions)
        
            if step % 50 == 0:
                print(f'Training loss: {float(loss_value):.3f} at step: {step}')
                #add to see mean training loss...?
        
        #dispay metrics
        display_train_metric = train_metric.result()
        train_metric.reset_state()
        print(f'Training accuracy over epoch: {float(display_train_metric):.3f}')
        #reset traiing metrics

        #run validation loop
        for X_batch_valid, y_batch_valid in valid_dataset:
            validation_predictions = model(X_batch_valid, training=False)
            valid_metric.update_state(y_batch_valid, validation_predictions)
            display_valid_metric = valid_metric.result()
        print(f'Validity for epoch: {float(display_valid_metric):.5f}')
        valid_metric.reset_state()
        #v_loss = loss_function(y_valid, validation_predictions)


    print('------- -- End of epochs -- -------')
    print('Training loss:', loss_value.numpy())
            #print('Training loss:', v_loss.numpy())

    ##running model on test data
    predict = model(X_test, training=False).numpy()
    compressor = predict[:, 0]
    turbine = predict[:, 1]

    print(compressor, turbine)

    mean_sq_error_compressor = np.mean((y_test[:, 0] - predict[:, 0]) **2)
    mean_sq_error_turbine = np.mean((y_test[:, 1] - predict[:, 1]) ** 2)

    if mean_sq_error_compressor <= (5 * (10**-8)):
        print(f'Decay compressor coefficient passed! Value: {mean_sq_error_compressor:.9f}')
    else:
        print(f'Decay compressor coefficient did not pass... Value: {mean_sq_error_compressor:.9f}')

    if mean_sq_error_turbine <= (1.5 * (10**-8)):
        print(f'Decay turbine coefficient passed! Value: {mean_sq_error_turbine:.9f}')
    else:
        print(f'Decay turbine coefficient did not pass... Value: {mean_sq_error_turbine:.9f}')


def main():
    X_features, y_coefficients = load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = prep_data(X_features, y_coefficients)

    model = model_architecture()
    run_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    
if __name__ == "__main__":
    main()
