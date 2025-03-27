import pandas as pd 
import numpy as np
import os 

# Load the data
data_path = '/Users/hoangthuyduongvu/Desktop/tumor-augmentation/data/train'
real = []
fake = []

for root, dirs, _ in os.walk(data_path):
    for d in dirs:
        if d == 'REAL':
            real.extend([os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d))])
        elif d == 'FAKE':
            fake.extend([os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d))])

#print("Number of real images:", len(real))
#print("Number of fake images:", len(fake))

# Create a DataFrame
data = pd.DataFrame({'real': real, 'fake': fake})
data.to_csv('data.csv', index=False)