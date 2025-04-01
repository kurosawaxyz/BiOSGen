import pandas as pd 
import numpy as np
import os 

# Load the data
data_path = 'demo/img'
real = []
fake = []

for root, _, f in os.walk(data_path):
    for d in f:
        print(d)
        if "HE" in d:
            real.append(os.path.join(root, d))
        else: 
            fake.append(os.path.join(root, d))

#print("Number of real images:", len(real))
#print("Number of fake images:", len(fake))

# Create a DataFrame
data = pd.DataFrame({'real': real, 'fake': fake})
data.to_csv('data.csv', index=False)