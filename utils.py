import torch
import numpy as np

from config import config

# parameters
# y: forward result (batch_size, # of classes)
# t: target (batch_size, )
# alpha: parameter
def reranking(y, t, alpha):
    batch_y_norm = np.zeros((config.BATCH_SIZE, config.num_classes))
    for idx,data in enumerate(y.data):
        value= torch.max(data)
        y.data[idx][t[idx]] = alpha * value          # rerank
        y_norm = y.data[idx] / torch.sum(y.data[idx])
        #print(y_norm.size())
        batch_y_norm[idx,:]=y_norm
    return torch.tensor(batch_y_norm, dtype=torch.float32)



def map_label_to_target(labels, t):
    batch_size = config.BATCH_SIZE
    targets = []
    for i in range(batch_size):
        targets.append(t[labels.data[i].item()])
    return targets


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
