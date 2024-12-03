import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Define a list of column names related to health and medical data
column_names = [
    "systolic blood pressure",  # Systolic blood pressure measurement
    "tobacco: cumulative tobacco (kg)",  # Cumulative tobacco usage in kilograms
    "low density lipoprotein cholesterol",  # LDL cholesterol level
    "adiposity",  # Adiposity or fatness of the body
    "famhist: family history of heart disease (Present, Absent)",  # Family history of heart disease
    "typea: type-A behavior",  # Type-A personality traits
    "obesity",  # Obesity indicator
    "alcohol: current alcohol consumption",  # Current alcohol consumption status
    "age: age at onset"  # Age at onset of a health condition (e.g., disease)
]

# Initialize the SentenceTransformer model with a pre-trained BERT model
# 'bert-base-nli-stsb-mean-tokens' is a BERT-based model fine-tuned for sentence embeddings
embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

# Use the embedder to convert the list of column names into embeddings (numeric representations)
# The embeddings will capture semantic meaning of each column name
column_name_embeddings = embedder.encode(column_names)

# Initialize the KMeans clustering model to group the column names into 2 clusters
clustering_model = KMeans(n_clusters=2)

# Fit the KMeans model on the column name embeddings to create the clusters
clustering_model.fit(column_name_embeddings)

# Retrieve the cluster labels (i.e., which cluster each column name belongs to)
cluster_assignment = clustering_model.labels_

# Print the cluster assignments as a NumPy array
# This shows which column name is assigned to which cluster (0 or 1)
np.save('data/saheart_clusters.npy', cluster_assignment)  


