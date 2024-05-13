import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn 
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def plot_save_heatmap(df_heatmap):
    le = LabelEncoder()
    df_heatmap['Records'] = le.fit_transform(df_heatmap['Records'])
    df_heatmap['Entity'] = le.fit_transform(df_heatmap['Entity'])
    df_heatmap['Organization type'] = le.fit_transform(df_heatmap['Organization type'])
    df_heatmap['Method'] = le.fit_transform(df_heatmap['Method'])
    
    # Generate the heatmap
    plt.figure(figsize=(10,5))
    seaborn.heatmap(df_heatmap[['Year', 'Records','Entity', 'Organization type', 'Method']].corr(), cmap='Spectral_r', annot=True)
    
    current_directory = os.getcwd()
    # Save the heatmap as an image
    plt.savefig(os.path.join(current_directory,'images', 'heatmap.png'))
    
    # Close the plot to free up resources
    plt.close()


def perform_pca(X):
    # Perform PCA and plot the cumulative explained variance ratio.

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    # Creating a DataFrame for the principal components
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Explained variance ratio
    print ( "Components = ", pca.n_components_ , ";\nTotal explained variance = ",
      round(pca.explained_variance_ratio_.sum(),5)  )
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs. Number of Principal Components')

    current_directory = os.getcwd()
    # Save the heatmap as an image
    plt.savefig(os.path.join(current_directory,'images', 'pca_explained_variance.png'))

    plt.close()



