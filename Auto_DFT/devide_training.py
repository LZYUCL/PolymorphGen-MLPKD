# Copyright (c) 2025 Zeyuan Li Wuhan University. Licensed under the MIT License.
# See the LICENSE file in the repository root for full details.
import dpdata
import numpy as np

# Load abacus/md format data
data = dpdata.LabeledSystem('./dataset', fmt = 'deepmd')

# Randomly select 100 indices for validation set; remaining indices for training set
index_validation = np.random.choice(len(data),size=100,replace=False)
index_training = list(set(range(len(data)))-set(index_validation))

# Create subsets: training set, validation set      
data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)

# Export training set and validation set (deepmd/npy format)                     
data_training.to_deepmd_npy('./training_data')
data_validation.to_deepmd_npy('./validation_data')

print('# Data contains %d frames' % len(data))
print('# Training data contains %d frames' % len(data_training))
print('# Validation data contains %d frames' % len(data_validation))
