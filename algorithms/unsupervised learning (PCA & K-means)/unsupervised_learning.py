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


def kmeans_pca_clustering(data, coeff_matrix, n_components=3, k=3, max_iters=300, random_state=42):

    np.random.seed(random_state)

    features = data[:, -13:]
    features_std = (features - np.mean(features, axis=0)) / np.std(features, axis=0, ddof=1)

    W = coeff_matrix[:, :n_components]
    Z = np.dot(features_std, W)

    n_samples = Z.shape[0]
    initial_idx = np.random.choice(n_samples, size=k, replace=False)
    centroids = Z[initial_idx].copy()

    # K-means Lloyd's algorithm
    cluster_labels = np.zeros(n_samples, dtype=int)
    for i in range(max_iters):
        # Euclidean distances
        distances = np.linalg.norm(Z[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        # Check convergence
        if np.array_equal(cluster_labels, new_labels):
            break
        cluster_labels = new_labels

        # Recompute centroids
        for i in range(k):
            assigned_points = Z[cluster_labels == i]
            if len(assigned_points) > 0:
                centroids[i] = np.mean(assigned_points, axis=0)

    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_counts = {int(cluster): int(count) for cluster, count in zip(unique_clusters, counts)}

    plt.figure(figsize=(9, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    x_data = Z[:, 0]
    y_data = Z[:, 1] if n_components > 1 else np.zeros_like(x_data)

    c_x = centroids[:, 0]
    c_y = centroids[:, 1] if n_components > 1 else np.zeros_like(c_x)

    for i in range(k):
        mask = (cluster_labels == i)
        plt.scatter(
            x_data[mask],
            y_data[mask],
            color=colors[i % len(colors)],
            alpha=0.75,
            edgecolors='none',
            s=45,
            label=f'Cluster {i} (n = {cluster_counts.get(i, 0)})'
        )

    #Plot Centroids
    plt.scatter(
        c_x,
        c_y,
        color="black",
        marker='X',
        s=180,
        linewidth=1.5,
        edgecolors='white',
        label='Centroids'
    )

    plt.title(f"K-Means (k={k}) on Top {n_components} Principal Components", fontsize=13, fontweight='bold')
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2" if n_components > 1 else "Fixed (1D)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()

    return cluster_labels, centroids, cluster_counts


def main():
    data = np.loadtxt("wine.txt")

    coeff_matrix, eigenvalues_sorted, eigenvectors = calculate_principal_component_coefficient_matrix(data)

    # visualize_data_objects(data, coeff_matrix, eigenvalues_sorted, eigenvectors)

    kmeans_pca_clustering(data, coeff_matrix, n_components=2, k=3, max_iters=300)

if __name__ == "__main__":
    main()




