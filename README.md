# %%
!pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab pyplotly

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# %%
df = pd.read_csv('single_genre_artists.csv')

# %%
# Drop unnecessary columns as they are not needed for clustering
df = df.drop(columns=['id_songs', 'name_song', 'id_artists', 'name_artists', 'genres', 'release_date'])

# %%
df.head()

# %%
# Define features for clustering based on the project description
features_for_clustering = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

# %%
# Separate the features from other columns
df_features = df[features_for_clustering].copy()

# %%
# Initialize StandardScaler to normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features)

# %%
# Create a DataFrame with the scaled features
df_scaled = pd.DataFrame(scaled_features, columns=features_for_clustering)

print("\nData preprocessing complete. Features have been normalized.")

# %%
# 2. Clustering Techniques - K-Means
# Determine the optimal number of clusters (k) using the Elbow Method
print("\nFinding the optimal number of clusters using the Elbow Method...")
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# %%
# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Errors)')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_method.png')
print("Elbow Method plot saved as 'elbow_method.png'.")

# %%
# Based on the plot, a good 'k' value will be where the elbow forms.
# Let's choose k=4 for this example, which is a common choice for this kind of data.
optimal_k = 4
print(f"\nOptimal number of clusters (k) chosen: {optimal_k}")

# %%
# Apply K-Means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(df_scaled)

# %%
# Add the cluster labels to the original DataFrame
df['cluster'] = cluster_labels
print("Cluster labels added to the original DataFrame.")


# %%
# 3. Cluster Evaluation and Interpretation
# Calculate evaluation metrics
silhouette_avg = silhouette_score(df_scaled, cluster_labels)
davies_bouldin_avg = davies_bouldin_score(df_scaled, cluster_labels)

print(f"\nEvaluation Metrics:")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_avg:.4f}")

# Interpret clusters by profiling their mean feature values
print("\nInterpreting clusters by profiling mean feature values...")
cluster_profiles = df.groupby('cluster')[features_for_clustering].mean()
print(cluster_profiles)

# %%
#4. Dimensionality Reduction (PCA) and Visualization
# Apply PCA for 2D visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['cluster'] = cluster_labels


# %%
# Create a bar chart showing average feature values per cluster
cluster_profiles_melted = cluster_profiles.reset_index().melt(
    id_vars='cluster',
    var_name='feature',
    value_name='average_value'
)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='feature',
    y='average_value',
    hue='cluster',
    data=cluster_profiles_melted,
    palette='viridis'
)
plt.title('Average Feature Values per Cluster')
plt.xlabel('Feature')
plt.ylabel('Average Value (Normalized)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cluster_feature_bar_chart.png')
print("Cluster feature bar chart saved as 'cluster_feature_bar_chart.png'.")

# %%
# Create a 2D scatter plot of the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='cluster',
    palette='viridis',
    data=df_pca,
    legend='full',
    alpha=0.6
)
plt.title('Song Clusters (PCA Reduced Dimensions)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('pca_cluster_plot.png')
print("PCA cluster plot saved as 'pca_cluster_plot.png'.")

# %%
# 5. Final Analysis and Export

# Export the final dataset with cluster labels
output_filename = 'final_clustered_data.csv'
df.to_csv(output_filename, index=False)
print(f"\nFinal clustered dataset with labels saved to '{output_filename}'.")



