# ===============================
# CUSTOMER SEGMENTATION PROJECT
# ===============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===============================
# STEP 1: LOAD DATASET
# ===============================

df = pd.read_csv("Mall_Customers.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ===============================
# STEP 2: DATA CLEANING
# ===============================

print("\nMissing Values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# ===============================
# STEP 3: EDA (VISUALIZATION)
# ===============================

plt.figure()
sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution")
plt.show()

plt.figure()
sns.histplot(df['Annual Income (k$)'], bins=20)
plt.title("Annual Income Distribution")
plt.show()

plt.figure()
sns.histplot(df['Spending Score (1-100)'], bins=20)
plt.title("Spending Score Distribution")
plt.show()

# ===============================
# STEP 4: SELECT FEATURES
# ===============================

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ===============================
# STEP 5: FEATURE SCALING
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# STEP 6: ELBOW METHOD
# ===============================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method (Optimal K)")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ===============================
# STEP 7: APPLY K-MEANS
# ===============================

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ===============================
# STEP 8: VISUALIZE CLUSTERS
# ===============================

plt.figure(figsize=(10,6))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=df,
    palette='Set1'
)

plt.title("Customer Segmentation (K-Means Clusters)")
plt.show()

# ===============================
# STEP 9: ANALYZE CLUSTERS
# ===============================

print("\nCluster Summary:")
print(df.groupby('Cluster').mean(numeric_only=True))

# ===============================
# FINAL MESSAGE
# ===============================

print("\n✅ Customer Segmentation Completed Successfully!")