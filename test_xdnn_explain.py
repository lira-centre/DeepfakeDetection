import xdnn_classification as xdnn
import numpy as np
import pandas as pd
from numpy import genfromtxt


model = xdnn.xDNNClassifier()

X_test = genfromtxt('data_df_X_test_deepfake_ffhq_finetuned.csv', delimiter=',')
y_test = pd.read_csv('data_df_y_test_deepfake_ffhq_finetuned.csv', delimiter=',',header=None)


print(X_test)

prototypes = model.load_model('xDNN_FFHQ_finetuned_weights')
print(list(prototypes['xDNNParms']['Parameters'][0]['Prototype'].values()))

names = list(prototypes['xDNNParms']['Parameters'][0]['Prototype'].values())
names.extend(list(prototypes['xDNNParms']['Parameters'][1]['Prototype'].values()))

arr_final = np.vstack((prototypes['xDNNParms']['Parameters'][0]['Centre'], prototypes['xDNNParms']['Parameters'][1]['Centre']))
print(len(arr_final))
print(len(names))


from scipy.spatial import distance

distances = []

for arr in arr_final:
    
    dst = distance.euclidean(arr.ravel(), X_test[0].ravel())
    distances.append(dst)

distances = np.array(distances)
print(distances.shape)
print(np.argpartition(distances,3)[:3])
closest_prototypes = np.argpartition(distances,3)[:3]

print(closest_prototypes)

for item in closest_prototypes:
    print(names[item])

print(y_test[0])




