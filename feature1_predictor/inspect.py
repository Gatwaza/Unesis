#!/usr/bin/env python3

import pandas as pd

dataset_path = '/Heart_Disease_Prediction.csv'  # Update the path as needed
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(df.head())
