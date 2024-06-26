import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Specify the folder where the file is located
data_folder = "C:/Users/Lenovo/OneDrive/Desktop/TASK_2/"  

# Load the Mall_Customers.csv file
file_path = os.path.join(data_folder, 'Mall_Customers.csv')
customers_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(customers_df.head())

# Display the summary of the dataset
print(customers_df.info())

# Selecting the relevant features for clustering
# For example, using 'Annual Income (k$)' and 'Spending Score (1-100)'
X = customers_df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Applying the K-means clustering algorithm
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Adding the cluster labels to the original dataframe
customers_df['Cluster'] = kmeans.labels_

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=customers_df, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
