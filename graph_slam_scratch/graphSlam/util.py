import numpy as np
def t2v(tr):
    # homogeneous transformation to vector
    v = np.zeros((3, 1))
    v[:2, 0] = tr[:2, 2]
    v[2] = np.arctan2(tr[1, 0], tr[0, 0])
    return v


def v2t(v):
    # vector to homogeneous transformation
    c = np.cos(v[2])
    s = np.sin(v[2])
    tr = np.array([[c, -s, v[0]],
                   [s, c, v[1]],
                   [0, 0, 1]])
    return tr
def id2index(value_id , num_params):
    return slice((num_params * value_id), (num_params * (value_id + 1)))