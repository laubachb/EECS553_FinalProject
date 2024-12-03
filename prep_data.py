import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prep_pima():
    # read the CSV file into a Pandas DataFrame
    data = pd.read_csv("diabetes.csv")
    X = data.iloc[:, :-1].values  # all rows, all but the last column
    y = data.iloc[:, -1].values   # all rows, only the last column (labels)
    print("Dataset size:", X.shape)
    
    # Separate positive and negative samples
    normal_samples = X[y == 0]  # Negative diagnoses
    anomal_samples = X[y == 1]  # Positive diagnoses
    print("Number of negative samples:", normal_samples.shape)
    print("Number of positive samples:", anomal_samples.shape)
    
    # Randomly shuffle negative samples and divide into train, validate, and test
    idx = np.random.permutation(normal_samples.shape[0])  
    normal_samples = normal_samples[idx]
    num_train = len(normal_samples) // 2   # split the training set in half
    x_train = normal_samples[:num_train]    

    np.random.shuffle(normal_samples)
    num_train = len(normal_samples) // 2
    x_train = normal_samples[:num_train]
    
    num_validate = len(normal_samples[num_train:]) // 2
    x_validate = normal_samples[num_train:num_train + num_validate]
    x_test = normal_samples[num_train + num_validate:]
    
    # Combine and scale the train + validate samples
    x_in = np.concatenate([x_train, x_validate])
    scaler = MinMaxScaler().fit(x_in)
    x_in = scaler.transform(x_in)
    
    # Prepare test data
    x_test = np.concatenate([x_test, anomal_samples])
    x_test = scaler.transform(x_test)
    
    # Split the scaled data back into train and validate
    x_train = x_in[:len(x_train)]
    y_train = np.zeros(len(x_train))
    
    x_valid = x_in[len(x_train):]
    y_valid = np.zeros(len(x_valid))
    
    # Prepare test labels
    y_test = np.concatenate([np.zeros(len(x_test) - len(anomal_samples)), np.ones(len(anomal_samples))])
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

# Call the refactored function
x_train, y_train, x_valid, y_valid, x_test, y_test = prep_pima()
