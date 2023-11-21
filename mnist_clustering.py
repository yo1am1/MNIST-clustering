import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
(x_train, y_train), (_, _) = mnist.load_data()

# Flatten images and normalize
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0

# Standardize data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)


# Additional task: Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    cluster_labels = []
    for i in range(10):
        true_labels = y_true[y_pred == i]
        if len(true_labels) > 0:
            most_common_label = np.argmax(np.bincount(true_labels))
            cluster_labels.append(most_common_label)

    correct_predictions = np.sum(y_pred == np.array(cluster_labels)[y_pred])
    accuracy = correct_predictions / len(y_true)
    return accuracy


# Build k-means clustering model
k_values = [10, 12, 14, 16, 18, 20]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    y_pred = kmeans.fit_predict(x_train_scaled)

    # Visualize some samples from each cluster
    fig, axs = plt.subplots(k, 10, figsize=(15, 8))
    plt.suptitle(f"K-Means Clustering with k={k}")

    for i in range(k):
        cluster_samples = x_train[y_pred == i][:10]
        for j in range(10):
            axs[i, j].imshow(cluster_samples[j], cmap="gray")
            axs[i, j].axis("off")

    # Calculate and print accuracy
    accuracy = calculate_accuracy(y_train, y_pred)
    print(f"Accuracy for k={k}: {accuracy:.2%}")

    plt.show()
