import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

dataset = 'pima' # saheart or pima 
description = True

if dataset == 'saheart':
    # Define a list of column names related to health and medical data for the SAHeart dataset 
    print('Generating cluster assignemtns for the SAHeart dataset')
    column_names = [
        "systolic blood pressure",  
        "tobacco: cumulative tobacco (kg)", 
        "low density lipoprotein cholesterol",  
        "adiposity", 
        "famhist: family history of heart disease (Present, Absent)",  
        "typea: type-A behavior",  
        "obesity",  
        "alcohol: current alcohol consumption", 
        "age: age at onset"]
    if description == True:
        column_names.insert(0, "The task is to detect if the South African has a coronary heart disease.")

elif dataset == 'pima':
    # Define a list of column names related to health and medical data for the PIMA dataset
    print('Generating cluster assignemtns for the PIMA dataset')
    column_names = [
        "number of times pregnant",  
        "plasma glucose concentration after 2 hours in an oral glucose tolerance test",  
        "diastolic blood pressure",  
        "triceps skin fold thickness",  
        "2-hour serum insulin",  
        "body mass index",  
        "diabetes pedigree function",  
        "age" 
        ]

# Initialize the SentenceTransformer model with a pre-trained BERT model
embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

# Use the embedder to convert the list of column names into embeddings (numeric representations)
column_name_embeddings = embedder.encode(column_names)

# Initialize the KMeans clustering model to group the column names into 2 clusters
clustering_model = KMeans(n_clusters=2)

# Fit the KMeans model on the column name embeddings to create the clusters
clustering_model.fit(column_name_embeddings)

# Retrieve the cluster labels (i.e., which cluster each column name belongs to)
cluster_assignment = clustering_model.labels_

# Print and save the cluster assignments as a NumPy array
print(cluster_assignment)

if dataset == 'saheart':
    if description == True:
        np.save('data/saheart_clusters_description.npy', cluster_assignment)  
    else:
        np.save('data/saheart_clusters.npy', cluster_assignment)  

if dataset == 'pima':
    np.save('pima_data/pima_clusters.npy', cluster_assignment)