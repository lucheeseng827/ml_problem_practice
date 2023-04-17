# Data Science Algorithms

## List of machine learning algorithms


| Algorithm | Benefit | Downside | Type | Supervised/Unsupervised |
| --- | --- | --- | --- | --- |
| Linear Regression | Simple and easy to implement | Assumes a linear relationship between the features and the response variable | Regression | Supervised |
| Logistic Regression | Provides class probabilities and is interpretable | Assumes a linear relationship between the features and the log-odds of the response variable | Classification | Supervised |
| Decision Trees | Easy to interpret and visualize | Can be prone to overfitting | Classification or Regression | Supervised |
| Random Forest | Reduces overfitting compared to decision trees and can handle missing values | Can be slower to train and less interpretable | Classification or Regression | Supervised |
| Naive Bayes | Simple and fast, can handle large feature sets | Assumes that all features are independent | Classification | Supervised |
| Support Vector Machines (SVM) | Effective in high-dimensional spaces and can handle non-linear decision boundaries | Can be prone to overfitting and requires careful selection of kernel function and hyperparameters | Classification or Regression | Supervised |
| K-Nearest Neighbors (KNN) | Non-parametric and flexible, can handle non-linear decision boundaries | Can be sensitive to the choice of k and the distance metric | Classification or Regression | Supervised |
| K-Means | Simple and efficient, can handle large datasets and easy to interpret | Requires pre-specification of the number of clusters, sensitive to initialization, and can get stuck in local optima | Clustering | Unsupervised |
| Hierarchical Clustering | Can create hierarchical tree-like cluster structure that can be visually inspected | Computationally expensive for large datasets | Clustering | Unsupervised |
| Principal Component Analysis (PCA) | Can reduce the dimensionality of the data while preserving as much of the variation as possible | Assumes a linear relationship between the features and can be sensitive to outliers | Dimensionality Reduction | Unsupervised |
| t-SNE | Can create a visual representation of high-dimensional data by mapping it to a lower-dimensional space while preserving local similarities | Can be computationally expensive for large datasets and may distort global structure | Dimensionality Reduction | Unsupervised |
| SVM | Good performance on high-dimensional data, ability to handle nonlinear data with kernel trick | Computationally intensive with large datasets, sensitive to the choice of kernel function | Classification | Supervised |
| Random Forest | Good performance on high-dimensional data, able to handle missing data, easy to interpret | Can overfit, computationally expensive with large datasets | Classification | Supervised |
| Gradient Boosting | Good performance on high-dimensional data, able to handle missing data, often achieves state-of-the-art performance | Can overfit, computationally expensive with large datasets | Classification, Regression | Supervised |
| K-Nearest Neighbors | Simple to understand and implement, no assumptions about the data distribution | Computationally intensive with large datasets, sensitive to the choice of k value | Classification, Regression | Supervised |
| Naive Bayes | Simple to understand and implement, computationally efficient, works well with high-dimensional data | Assumes independence of features, may not work well with correlated features | Classification | Supervised |
| DBSCAN | Able to handle non-linear data, no need to specify number of clusters | Sensitive to the choice of hyperparameters, does not work well with data with varying densities | Clustering | Unsupervised |
| Hierarchical Clustering | No need to specify number of clusters, produces a dendrogram to visualize clusters | Computationally intensive, sensitive to the choice of distance metric | Clustering | Unsupervised |
| PCA | Able to handle high-dimensional data, reduces dimensionality while retaining important information | May lose some information in the data, may be sensitive to outliers | Dimensionality reduction | Unsupervised |
| K-Means | Computationally efficient, simple to implement | Sensitive to the choice of k value, does not work well with non-linear data | Clustering | Unsupervised |



## List of dimensionality reduction algorithms

### Autoencoders
Autoencoders are neural network-based algorithms that can learn a compressed representation of high-dimensional data. They work by learning to encode the input data into a lower-dimensional latent space, and then decoding it back to the original dimensions. Autoencoders can be used for both supervised and unsupervised tasks.

### t-SNE
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction algorithm that is particularly useful for visualizing high-dimensional data. It works by modeling the similarity between points in the high-dimensional space and the low-dimensional space, and minimizing the divergence between the two.

### UMAP
UMAP (Uniform Manifold Approximation and Projection) is a newer non-linear dimensionality reduction algorithm that is similar to t-SNE, but can scale better to larger datasets. It works by preserving the local structure of the data in the low-dimensional space, while also allowing for global structure to be retained.

### PCA-based methods
There are several dimensionality reduction methods that are based on Principal Component Analysis (PCA), including Sparse PCA, Incremental PCA, and Kernel PCA. These methods work by finding the principal components of the data and projecting it onto a lower-dimensional space.

### Random Projection
Random Projection is a simple but effective dimensionality reduction technique that works by projecting high-dimensional data onto a random subspace of lower dimensions. The method is computationally efficient and can work well for sparse data.

### Independent Component Analysis (ICA)
ICA is a method for separating a multivariate signal into independent, non-Gaussian components. It can be used for both signal processing and dimensionality reduction, and has applications in image processing and bioinformatics.

### Linear Discriminant Analysis (LDA)
LDA is a supervised dimensionality reduction algorithm that works by finding the linear combinations of features that best separate the classes in the data. It can be used for classification tasks, and is particularly useful when there are many more features than observations.

### Multi-Dimensional Scaling (MDS)
MDS is a method for visualizing high-dimensional data by finding a low-dimensional representation that preserves the pairwise distances between points. It can be used for both supervised and unsupervised tasks, and has applications in social sciences and marketing research.
