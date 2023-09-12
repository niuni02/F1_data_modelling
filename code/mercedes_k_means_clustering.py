# Mercedes K Means Clustering
# Data source: https://github.com/oracle-devrel/redbull-pit-strategy/tree/main

# ***** PRE-REQUISITES *****
from gettext import install
import os
cwd = os.getcwd() # current working directory
print("Current working directory: {0}".format(cwd))
# Set the desired directory path
path = '/Users/niuni/Desktop/work/GitHub/F1_data_modelling/raw_data'
# Change the working directory to the desired one
os.chdir(path)
# Check the working directory is set
cwd = os.getcwd()
# Print the current working directory
print("Current working directory: {0}".format(cwd))

# ***** IMPORT LIBRARIES *****
from gettext import install
import pandas as pd
import pip
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ***** IMPORT CSV *****
data = pd.read_csv("final_data.csv")
data = data[data['Team'] == 'Mercedes']
x = data['lapNumberAtBeginingOfStint']
y = data['designedLaps']
x = x.to_numpy().reshape(-1, 1)
y = y.to_numpy().reshape(-1, 1)

# ***** K-MEANS CLUSTERING *****
# Number of clusters (K)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(x,y)
# Cluster assignments for each data point
labels = kmeans.labels_
# Cluster centers
centers = kmeans.cluster_centers_
plt.scatter(data['lapNumberAtBeginingOfStint'], data['designedLaps'], c=labels, cmap='viridis')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], c='red', marker='X', s=200)
plt.title(f'K-Means Clustering with {n_clusters} Clusters')
plt.xlabel('lapNumberAtBeginingOfStint')
plt.ylabel('designedLaps')
plt.show()

