from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt

# Create a sample dissimilarity matrix
dissimilarity_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

# Initialize MDS with default parameters
mds = MDS(n_components=2, random_state=42)

# Compute the MDS embedding
embedding = mds.fit_transform(dissimilarity_matrix)
print(embedding)

# Plot the results
plt.scatter(dissimilarity_matrix[:, 0], dissimilarity_matrix[:, 1])
plt.title("dissimalirity")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title("MDS Embedding")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
