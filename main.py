import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('THE World University Rankings 2016-2026.csv')

# Select relevant columns
feature_cols = ['Student Population', 'Students to Staff Ratio', 'International Students', 'Female to Male Ratio', 'Teaching', 'Research Environment', 'Research Quality', 'Industry Impact', 'International Outlook']
meta_cols = ['Name', 'Rank', 'Overall Score']

# Create a clean dataframe with no missing values in feature columns
df_clean = df[df['Year'] == 2026].dropna(subset=feature_cols).copy()
df_clean['Student Population'] = np.log10(df_clean['Student Population'].str.replace(',', '').astype(float) + 1)
df_clean['Students to Staff Ratio'] = np.log10(df_clean['Students to Staff Ratio'] + 1)
df_clean['International Students'] = df_clean['International Students'].str.rstrip('%').replace('', 0).astype(float)
df_clean['Female to Male Ratio'] = df_clean['Female to Male Ratio'].str.split(':').str[0].replace('', 0).astype(float)

# Extract features
X = df_clean[feature_cols]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(X_pca);

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df = pd.concat([pca_df, df_clean[meta_cols].reset_index(drop=True)], axis=1)

# Scree plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.savefig('scree_plot.png')

# Biplot (Scatter plot of PC1 vs PC2)
plt.figure(figsize=(12, 10))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue='Rank', palette='viridis_r', alpha=0.8, edgecolor=None)
plt.title('PCA: PC1 vs PC2')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

# Add loading vectors to the plot
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
for i, feature in enumerate(feature_cols):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='r', alpha=0.5, head_width=0.1) # Scale arrows for visibility
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.5, feature, color='r', ha='center', va='center')

plt.grid(True)
plt.savefig('pca_biplot.png')

# Display explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Display loadings
loadings_df = pd.DataFrame(pca.components_.T, index=feature_cols, columns=[f'PC{i+1}' for i in range(len(feature_cols))])
print("\nLoadings (Eigenvectors):")
print(loadings_df)