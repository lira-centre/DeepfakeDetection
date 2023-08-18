import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
from sklearn.manifold import TSNE

x_axis_labels = ['Original','Fake']
label_names = {0: "Original train", 1: "Fake train", 2: "Original test", 3: "Fake test"}
label_names_2 = {0: "Original", 1: "Fake"}

X_train = genfromtxt('data_df_X_train_deepfake_imagenet_finetuned.csv', delimiter=',')
y_train = pd.read_csv('data_df_y_train_deepfake_imagenet_finetuned.csv', delimiter=',',header=None)
X_test = genfromtxt('data_df_X_test_deepfake_imagenet_finetuned.csv', delimiter=',')
y_test = pd.read_csv('data_df_y_test_deepfake_imagenet_finetuned.csv', delimiter=',',header=None)
print("finish read")

pd_y_train_labels = y_train[1]
pd_y_test_labels = y_test[1]

pd_y_train_images = y_train[0]
pd_y_test_images = y_test[0]

y_train_labels = pd_y_train_labels.to_numpy()
y_test_labels = pd_y_test_labels.to_numpy()

y_train_images = pd_y_train_images.to_numpy()
y_test_images = pd_y_test_images.to_numpy()


#y_test_labels = y_test_labels + 2

all_data = np.concatenate((X_train, X_test), axis=0)
all_labels = np.concatenate((y_train_labels, y_test_labels), axis=0)

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(X_train)

# Create a scatter plot with colors based on labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_train_labels, cmap='viridis', s=1)
plt.title("t-SNE Visualization for training samples of pre-trained features on image-to-image Deepfake ImageNet subset",
          fontsize=14)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)

# Create a legend based on label names
legend_labels = [label_names_2[label] for label in np.unique(all_labels)]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Label Names")

plt.colorbar(label="Label")
plt.show()



tsne_2 = TSNE(n_components=2)
tsne_result = tsne.fit_transform(X_test)

# Create a scatter plot with colors based on labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_test_labels, cmap='viridis', s=1)
plt.title("t-SNE Visualization for testing samples of pre-trained features on image-to-image Deepfake ImageNet subset",
           fontsize=14)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)

# Create a legend based on label names
legend_labels = [label_names_2[label] for label in np.unique(all_labels)]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Label Names")
plt.colorbar(label="Label")
plt.show()