import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    


class FCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, drop=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class SwiGLU(nn.Module):
    def __init__(self, d_model, swiglu_ratio=8/3):
        super().__init__()
        self.WG = nn.Linear(d_model, int(d_model * swiglu_ratio))
        self.W1 = nn.Linear(d_model, int(d_model * swiglu_ratio))
        self.W2 = nn.Linear(int(d_model * swiglu_ratio), d_model)
    
    def forward(self, x):
        g = F.silu(self.WG(x))
        z = self.W1(x)
        return self.W2(g * z)
    
    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()
    
    

class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, SE_dim, device, drop=0.,
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim

        self.value_embedding = nn.Linear(feature_dim, embed_dim, bias=True, device=device)
        self.position_encoding = PositionalEncoding(embed_dim)
        self.daytime_embedding = nn.Embedding(1440, embed_dim, device=device)
        self.weekday_embedding = nn.Embedding(7, embed_dim, device=device)

        self.spatial_embedding = nn.Linear(SE_dim, embed_dim, device=device)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, spa_mx=None):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)

        de = self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * 1440).round().long())
        x += de
        we = self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x += we
            
        x += self.spatial_embedding(spa_mx).unsqueeze(0).unsqueeze(0)
        x = self.dropout(x)
        
        return x
    



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


    
def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))



def cluster_regions(distance_matrix, target_clusters, balance, tolerance=1.0):
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square (NxN)!")
    
    condensed_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    linkage_matrix = linkage(condensed_dist, method='average')
    labels = fcluster(linkage_matrix, target_clusters, criterion='maxclust')

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    if not balance:
        return [points for points in clusters.values()]
    
    avg_size = len(distance_matrix) // target_clusters
    max_size = int(avg_size * (1 + tolerance))
    
    for label in list(clusters.keys()):
        while len(clusters[label]) > max_size:
            point_to_move = clusters[label].pop()
            min_label = None
            min_distance = float('inf')
            for other_label, other_points in clusters.items():
                if other_label != label and len(other_points) < max_size:
                    avg_dist = np.mean([distance_matrix[point_to_move, p] for p in other_points])
                    if avg_dist < min_distance:
                        min_distance = avg_dist
                        min_label = other_label
            if min_label is not None:
                clusters[min_label].append(point_to_move)

    balanced_clusters = [points for points in clusters.values()]
    return balanced_clusters




def hierarchical_clustering(distance_matrix, cluster_targets, balance=True):
    results = []
    current_matrix = distance_matrix
    
    for target in cluster_targets:
        clusters = cluster_regions(current_matrix, target, balance=balance)
        results.append(clusters)
        new_matrix = np.zeros((target, target))
        for i, group_i in enumerate(clusters):
            for j, group_j in enumerate(clusters):
                distances = [
                    distance_matrix[point_i, point_j]
                    for point_i in group_i
                    for point_j in group_j
                ]
                new_matrix[i, j] = np.mean(distances)
        current_matrix = new_matrix
    
    return results

    

        
        