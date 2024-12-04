import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_dataset(dataset, train_split = 0.50, validation_split = 0.25): # predefined splits according to paper 
    if dataset == 'saheart':
        print('Removing text-based descriptors from SAHeart dataset')
        df = pd.read_csv("saheart_data/saheart.txt", skiprows=14, header=None)
        # Replace 'Present' with 1 and 'Absent' with 0 in the DataFrame
        df.replace({'Present': 1, 'Absent': 0}, inplace=True)
    else:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv("pima_data/diabetes.csv")

    # Separate features (X) and target (y)
    X = df.iloc[:, :-1].values  # All columns except the last one (features)
    y = df.iloc[:, -1].values   # Last column (target/labels)
    print(f'Datatset size: {X.shape}')

    # Separate positive and negative samples
    normal_data = X[y == 0]  # Negative diagnoses
    anomalies = X[y == 1]  # Positive diagnoses
    
    # randomly shuffle data 
    idx = np.random.permutation(normal_data.shape[0]) 
    normal_data = normal_data[idx] 

    # split dataset into train, validate test 
    num_train = int(train_split * len(normal_data))
    x_train = normal_data[:num_train]    
    num_validate = int(validation_split * len(normal_data))
    x_validate = normal_data[num_train:][:num_validate]
    x_test = normal_data[num_train:][num_validate:]

    # Combine and scale the train + validate samples
    x_train_validate = np.concatenate([x_train, x_validate])
    scaler = MinMaxScaler().fit(x_train_validate)
    x_train_validate = scaler.transform(x_train_validate)
    
    # Split the scaled data back into train and validate
    x_train = x_train_validate[:len(x_train)]
    y_train = np.zeros(len(x_train))
    x_validate = x_train_validate[len(x_train):]
    y_validate = np.zeros(len(x_validate))
    
    # Prepare test labels
    x_test = np.concatenate([x_test, anomalies])
    x_test = scaler.transform(x_test)
    y_test = np.concatenate([np.zeros(len(normal_data[num_train:][num_validate:])), np.ones(len(anomalies))]) 
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test


