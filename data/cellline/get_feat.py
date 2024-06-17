import numpy as np
import pandas as pd 
import torch
from sequitur.models import LINEAR_AE
from sequitur.models import LSTM_AE
from sequitur import quick_train
from sklearn.preprocessing import StandardScaler
##########进行训练
mut_feature =  np.load('target_mut.npy')
ge_feature  =  np.load('target_ge.npy')
ppi         =  np.load('ppi.coo.npy')
nodes_ge    =  np.load('nodes_ge.npy')
nodes_mut   =  np.load('nodes_mut.npy')

def get_index(point,lst):
    x = point[0]
    y = point[1]
    x = np.argwhere(lst == x)[0][0]
    y = np.argwhere(lst == y)[0][0]
    return x,y

matrix_ge = np.zeros((3384, 3384), dtype='float32')
for p in ppi:
    if p[0] not in nodes_ge:
        continue
    if p[1] not in nodes_ge:
        continue
    index = get_index(p,nodes_ge)
    matrix_ge[index] = 1

matrix_mut = np.zeros((10707, 10707), dtype='float32')
for p in ppi:
    if p[0] not in nodes_mut:
        continue
    if p[1] not in nodes_mut:
        continue
    index = get_index(p,nodes_mut)
    matrix_mut[index] = 1

newge = np.dot(ge_feature,matrix_ge)
newmut = np.dot(mut_feature,matrix_mut)

newfeat = np.append(newge,newmut,axis=1)
scaler = StandardScaler().fit(newfeat)
newfeat = scaler.transform(newfeat)

encoder, decoder, _, losses = quick_train(LINEAR_AE, torch.from_numpy(newfeat).float(), encoding_dim=786, lr=1e-4, epochs=100,denoise=False)
ppife = encoder(torch.from_numpy(np.array(newfeat)).float())
print(losses)
np.save("oneil_cell_feat.npy", ppife.detach().numpy())