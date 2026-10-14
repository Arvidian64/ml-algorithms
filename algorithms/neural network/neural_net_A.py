import json
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
from keras import ops
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
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.50, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Standardiesera features och mål
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_val_scaled = scaler_x.transform(X_val)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Dubbelgrenat nätverk
inputs = layers.Input(shape=(X_train.shape[1],))

# Nätverk för Compressor Decay
c = layers.Dense(256, activation="swish")(inputs)
c = layers.Dense(256, activation="swish")(c)
c_res = layers.Dense(128, activation="swish")(c)
c = layers.Dense(128, activation="swish")(c_res)
c = layers.Add()([c, c_res])
c = layers.Dense(64, activation="swish")(c)
comp_out = layers.Dense(1, activation="linear", name="comp_out")(c)

# Nätverk för Turbine Decay
t = layers.Dense(256, activation="swish")(inputs)
t = layers.Dense(256, activation="swish")(t)
t_res = layers.Dense(128, activation="swish")(t)
t = layers.Dense(128, activation="swish")(t_res)
t = layers.Add()([t, t_res])
t = layers.Dense(64, activation="swish")(t)
turb_out = layers.Dense(1, activation="linear", name="turb_out")(t)

# Kombinera tillbaka till nätverk (batch, 2)
outputs = layers.Concatenate()([comp_out, turb_out])
model = keras.Model(inputs=inputs, outputs=outputs)

class CustomAdam(keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-12,
        amsgrad=False,
        nesterov=True,
        weight_decay=0.0,
        name="CustomAdam",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.nesterov = nesterov

    def build(self, variables):
        super().build(variables)
        self._m = [
            self.add_variable_from_reference(var, "m") for var in variables
        ]
        self._v = [
            self.add_variable_from_reference(var, "v") for var in variables
        ]
        if self.amsgrad:
            self._v_max = [
                self.add_variable_from_reference(var, "v_max") for var in variables
            ]

    def update_step(self, gradient, variable, learning_rate):
        idx = self._get_variable_index(variable)
        m = self._m[idx]
        v = self._v[idx]

        lr = ops.cast(learning_rate, variable.dtype)
        beta_1 = ops.cast(self.beta_1, variable.dtype)
        beta_2 = ops.cast(self.beta_2, variable.dtype)
        eps = ops.cast(self.epsilon, variable.dtype)
        step = ops.cast(self.iterations + 1, variable.dtype)

        # Uppdatera jäviga första och andramoments-uppskattningar
        self.assign(m, beta_1 * m + (1.0 - beta_1) * gradient)
        self.assign(v, beta_2 * v + (1.0 - beta_2) * ops.square(gradient))

        beta_1_power = ops.power(beta_1, step)
        beta_2_power = ops.power(beta_2, step)

        # Beräkna jäv-rättade moment
        m_hat = m / (1 - tf.pow(beta_1, step))

        if self.amsgrad:
            v_max = self._v_max[idx]
            self.assign(v_max, ops.maximum(v_max, v))
            v_corr = v_max / (1.0 - ops.power(beta_2, step))
        else:
            v_corr = v / (1.0 - ops.power(beta_2, step))

        # nesterov lookahead
        if self.nesterov:
            m_update = beta_1 * m_hat + (1.0 - beta_1) / (1.0 - beta_1_power) * gradient
            m_hat = m_update

        # Uppdatera parametrar
        denom = ops.sqrt(v_corr) + eps
        self.assign_sub(variable, lr * m_hat / denom)


class TargetEvaluationCheckpoint(keras.callbacks.Callback):
    def __init__(self, X_val_scaled, y_val, scaler_y, comp_target=5.0e-8, turb_target=1.5e-8, eval_freq=10):
        super().__init__()
        self.X_val_scaled = X_val_scaled
        self.y_val = y_val.astype(np.float64)
        self.scaler_y = scaler_y
        self.comp_target = comp_target
        self.turb_target = turb_target

        self.eval_freq = eval_freq
        self.best_combined_score = float("inf")
        self.best_weights = None
        self.targets_met = False

    def on_epoch_end(self, epoch, logs=None):
        # Predict on validation data and inverse-transform to physical units
        if (epoch + 1) % self.eval_freq != 0:
            return

        preds_scaled = self.model.predict(self.X_val_scaled, verbose=0)
        preds = self.scaler_y.inverse_transform(preds_scaled).astype(np.float64)

        comp_mse = mean_squared_error(self.y_val[:, 0], preds[:, 0])
        turb_mse = mean_squared_error(self.y_val[:, 1], preds[:, 1])

        combined_score = (comp_mse / self.comp_target) + (turb_mse / self.turb_target)

        # Track best turbine performance
        if combined_score < self.best_combined_score:
            self.best_combined_score = combined_score
            self.best_weights = self.model.get_weights()
            print(f"\n[Epoch {epoch + 1}] New Best -> Comp MSE: {comp_mse:.3e} | Turb MSE: {turb_mse:.3e}")

        # Check if both thresholds are satisfied
        if comp_mse <= self.comp_target and turb_mse <= self.turb_target:
            if not self.targets_met:
                print(f"\n>>> Both targets satisfied at epoch {epoch + 1}!")
                self.best_weights = self.model.get_weights()
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            print("\nRestoring weights from best turbine performance epoch...")
            self.model.set_weights(self.best_weights)



optimizer = CustomAdam(learning_rate=0.0016, beta_1=0.9, beta_2=0.9999, epsilon=1e-12, amsgrad=False, nesterov=True, weight_decay=1e-12)

# kompilera modellen
model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mse"]
)

# Träning

callbacks = [
    TargetEvaluationCheckpoint(
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        scaler_y=scaler_y,
        comp_target=5.0e-8,
        turb_target=1.5e-8
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.7,
        patience=100,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=200,
        restore_best_weights=False,
        verbose=1
    )
]

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=1000,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# Utvärdering

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mse_compressor = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_turbine = mean_squared_error(y_test[:, 1], y_pred[:, 1])

target_mse_comp = 5.0e-8
target_mse_turb = 1.5e-8

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION (UNSCALED)")
print("=" * 60)
print(f"Compressor Decay MSE: {mse_compressor:.6e} (Target: <= {target_mse_comp:.1e}) -> "
      f"{'PASSED' if mse_compressor <= target_mse_comp else 'FAILED'}")
print(f"Turbine Decay MSE:    {mse_turbine:.6e} (Target: <= {target_mse_turb:.1e}) -> "
      f"{'PASSED' if mse_turbine <= target_mse_turb else 'FAILED'}")
print("=" * 60)

print("\nSample Predictions vs Ground Truth:")
for i in range(min(5, len(y_test))):
    print(f"Sample {i+1} | "
          f"Comp Actual: {y_test[i, 0]:.6f}, Pred: {y_pred[i, 0]:.6f} | "
          f"Turb Actual: {y_test[i, 1]:.6f}, Pred: {y_pred[i, 1]:.6f}")



