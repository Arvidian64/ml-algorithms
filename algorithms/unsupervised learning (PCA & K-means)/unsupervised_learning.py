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


# Eigenvärdes-algoritm
V = [[1.0 if i == j else 0.0 for j in range(n_features)] for i in range(n_features)]
A = [row[:] for row in cov_matrix]

max_rotations = 200
for _ in range(max_rotations):
    # Find largest off-diagonal element
    max_val = 0.0
    p, q = 0, 1
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j

    if max_val < 1e-11:
        break

    # Calculate rotation angle
    diff = A[q][q] - A[p][p]
    if abs(A[p][q]) < 1e-15:
        t = 0.0
    else:
        theta = diff / (2.0 * A[p][q])
        t = math.copysign(1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0)), theta)

    c = 1.0 / math.sqrt(t * t + 1.0)
    s = t * c

    # Apply Givens rotation to A: A_new = G^T * A * G
    App = A[p][p]
    Aqq = A[q][q]
    Apq = A[p][q]

    A[p][p] = c * c * App - 2.0 * s * c * Apq + s * s * Aqq
    A[q][q] = s * s * App + 2.0 * s * c * Apq + c * c * Aqq
    A[p][q] = 0.0
    A[q][p] = 0.0

    for i in range(n_features):
        if i != p and i != q:
            a_ip = A[i][p]
            a_iq = A[i][q]
            A[i][p] = c * a_ip - s * a_iq
            A[p][i] = A[i][p]
            A[i][q] = s * a_ip + c * a_iq
            A[q][i] = A[i][q]

    # Accumulate transformations into eigenvector matrix V
    for i in range(n_features):
        v_ip = V[i][p]
        v_iq = V[i][q]
        V[i][p] = c * v_ip - s * v_iq
        V[i][q] = s * v_ip + c * v_iq

# Extract eigenvalues from diagonal
eigenvalues = [A[i][i] for i in range(n_features)]

#Sortera komponenter utefter eigenvärden
indices = list(range(n_features))
indices.sort(key=lambda idx: eigenvalues[idx], reverse=True)

coeff_matrix = [
    [V[i][component_idx] for component_idx in indices]
    for i in range(n_features)
]

#Projicera data för att beräkna nya "features"
# Z = X_std * CoeffMatrix
new_features = []
for row in X_std:
    transformed_row = [
        sum(row[j] * coeff_matrix[j][pc_idx] for j in range(n_features))
        for pc_idx in range(n_features)
    ]
    new_features.append(transformed_row)


def save_to_file(coeff_matrix, new_features):
    with open("pca_coefficients.txt", "w") as f:
        for row in coeff_matrix:
            f.write(" ".join(f"{val:12.6f}" for val in row) + "\n")

    # Save transformed scores for all objects
    with open("wine_pca_transformed.txt", "w") as f:
        for row in new_features:
            f.write(" ".join(f"{val:12.6f}" for val in row) + "\n")

    print("PCA completed successfully.")
    print(f"Saved 'pca_coefficients.txt' ({n_features}x{n_features})")
    print(f"Saved 'wine_pca_transformed.txt' ({n_samples}x{n_features})")


