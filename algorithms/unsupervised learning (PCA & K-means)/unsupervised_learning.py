import math
import numpy as np
import matplotlib.pyplot as plt


def calculate_principal_component_coefficient_matrix(data):
    X = data[:, -13:]

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    X_std = (X - mean) / std

    # Kovarians-matris
    cov_matrix = np.cov(X_std, rowvar=False)

    #Eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    #Sortering
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]

    coeff_matrix = eigenvectors[:, sorted_indices]

    transformed_features = np.dot(X_std, coeff_matrix)

    np.savetxt("pca_coefficients.csv", coeff_matrix, delimiter=",", fmt="%.6f")
    np.savetxt("wine_pca_transformed.csv", transformed_features, delimiter=",", fmt="%.6f")

    explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)

    print("explained variance ratio per component:")
    for i, ratio in enumerate(explained_variance_ratio, 1):
        print(f"{i}: {ratio:.4f} (cumulative: {np.sum(explained_variance_ratio[:i]):.4f})")

    return coeff_matrix, eigenvalues_sorted, eigenvectors

def visualize_data_objects(data, coeff_matrix, eigenvalues_sorted, eigenvectors):
    # Projicerar datan på de två viktigaste komponenterna och visualiserar den

    labels = data[:, 0].astype(int)
    X = data[:, -13:]

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    X_std = (X - mean) / std

    W_2d = coeff_matrix[:, :2]
    Z_2d = np.dot(X_std, W_2d)

    total_variance = np.sum(eigenvalues_sorted)
    var_pc1 = (eigenvalues_sorted[0] / total_variance) *100
    var_pc2 = (eigenvalues_sorted[1] / total_variance) *100

    plt.figure(figsize=(9, 6))

    unique_labels = np.unique(labels)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for idx, label in enumerate(unique_labels):
        mask = (labels == label)
        plt.scatter(
            Z_2d[mask, 0],
            Z_2d[mask, 1],
            label=f"Wine Class {label}",
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            alpha=0.85,
            edgecolors='k',
            linewidths=0.6,
            s=55
        )

    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.title("Wine dataset: 2D PCA Projection", fontsize=13, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({var_pc1:.2f}% Variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_pc2:.2f}% Variance)", fontsize=11)
    plt.grid(True, linestyle=":",alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


data = np.loadtxt("wine.txt")

coeff_matrix, eigenvalues_sorted, eigenvectors = calculate_principal_component_coefficient_matrix(data)

visualize_data_objects(data, coeff_matrix, eigenvalues_sorted, eigenvectors)




