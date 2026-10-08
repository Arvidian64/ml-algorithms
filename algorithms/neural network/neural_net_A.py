import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import keras
from keras import layers

with open("data_labels.json", "r") as read_file:
    lists = json.load(read_file)

features = lists[0]
decay_coefficients = lists[1]


data = np.loadtxt("maintenance.dat")

# Separera egenskaper (X) från mål-koefficienter (Y)
X = data[:, :len(features)]
y = data[:, len(features):]

# Splitta träning o test-data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=42
)

# Skala egenskaperna till en gemensam variabel
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential(
    [
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(y_train.shape[1], activation="linear")
    ]
)

# kompilera modellen
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mse"]
)

# Träning
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Utvärdering
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test loss:", test_loss)
print("Test mae:", test_mae)



# Prediktionsexempel
sample_preds = model.predict(X_test_scaled[:3])
print("\nPredictions vs Actual:")
for pred, actual in zip(sample_preds, y_test[:3]):
    print(f"Pred: {pred.round(4)} | Actual: {actual.round(4)}")

y_pred = model.predict(X_test_scaled)

# Kolumn 0: Compressor decay, Kolumn 1: Turbine decay
mse_compressor = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_turbine = mean_squared_error(y_test[:, 1], y_pred[:, 1])

print(f"Compressor Decay MSE: {mse_compressor}")
print(f"Turbine Decay MSE:    {mse_turbine}")



