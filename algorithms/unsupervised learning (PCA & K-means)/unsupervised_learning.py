import math

def read_wine():
    data = []

    with open('wine.txt') as file:
        for line in file:
            parts = line.split(" ")
            float_parts = []
            for i in parts:
                float_parts.append(float(i))
            data.append(float_parts[1:])

    return data

data = read_wine()

n_samples = len(data)

n_features = len(data[0])

# Standardisera features (Z-score)

# MedelKolumn
means = [0.0] * n_features
for row in data:
    for j in range(n_features):
        means[j] += row[j]
means = [m / n_samples for m in means]

# Standard deviation
stds = [0.0] * n_features
for row in data:
    for j in range(n_features):
        stds[j] += (row[j] - means[j]) ** 2
stds = [math.sqrt(s / n_samples) for s in stds]

# Standardiserad matris
X_std = []
for row in data:
    std_row = [(row[j] - means[j]) / stds[j] for j in range(n_features)]
    X_std.append(std_row)

# Ko-varians matris
cov_matrix = [[0.0] * n_features for _ in range(n_features)]
for i in range(n_features):
    for j in range(i, n_features):
        cov_val = sum(X_std[k][i] * X_std[j][i] for k in range(n_samples)) / (n_samples - 1)
        cov_matrix[i][j] = cov_val
        cov_matrix[j][i] = cov_val




