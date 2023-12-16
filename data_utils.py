import numpy as np
import torch
def split_noniid(train_data, alpha, n_clients):
    
    '''cifar10数据集需要'''
    targets = torch.tensor(train_data.targets)
    '''cifar10数据集需要'''

    n_classes = len(train_data.classes)
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    data_id = [i for i in range(len(train_data))]

    #class_idcs = [np.argwhere(train_data.targets[data_id]==y).flatten() for y in range(n_classes)]
    class_idcs = [np.argwhere(targets[data_id]==y).flatten() for y in range(n_classes)]
    #找到分类标签为0的数据index，将它们放在一起。
    #找到分类标签为1的数据index，将它们放在一起。
    #找到分类标签为9的数据index，将它们放在一起。
    #最后拉成一维，10个类别。
    #class_idcs: [tensor([    1,    21,    34,  ..., 59952, 59972, 59987]), tensor([    3,     6,     8,  ..., 59979, 59984, 59994]), tensor([    5,    16,    25,  ..., 59983, 59985, 59991])...
    #class_idcs[0]代表标签为0的数据的index。

    #设置包含n个空子列表的列表，client_idcs[0]表示客户端0拥有的数据，是一个包含分类标签为0的index的列表，client_idcs[1]表示客户端1拥有的数据的index.
    client_idcs = [[] for _ in range(n_clients)] 

    for c, fracs in zip(class_idcs, label_distribution):
        # tensor([    1,    21,    34,  ..., 59952, 59972, 59987])是分类标签为0的所有数据的index， 
        # [0.05100175 0.61256026 0.33643799]是n_clients个客户端拥有数据的比例，
        # 表示将上述数据，按照上述比例分配给n_clients个客户端。这里n_clients=3
        '''
        tensor([    1,    21,    34,  ..., 59952, 59972, 59987]) [0.05100175 0.61256026 0.33643799]
        tensor([    3,     6,     8,  ..., 59979, 59984, 59994]) [0.34615319 0.40352317 0.25032364]
        tensor([    5,    16,    25,  ..., 59983, 59985, 59991]) [0.83246995 0.12181871 0.04571134]
        tensor([    7,    10,    12,  ..., 59978, 59980, 59996]) [0.53285468 0.11237193 0.3547734 ]
        tensor([    2,     9,    20,  ..., 59943, 59951, 59975]) [0.22042591 0.17410638 0.60546772]
        tensor([    0,    11,    35,  ..., 59968, 59993, 59997]) [0.24424347 0.54340154 0.21235499]
        tensor([   13,    18,    32,  ..., 59982, 59986, 59998]) [0.48753972 0.28305517 0.22940512]
        tensor([   15,    29,    38,  ..., 59963, 59977, 59988]) [0.18207834 0.40070911 0.41721255]
        tensor([   17,    31,    41,  ..., 59989, 59995, 59999]) [0.1962258  0.41010856 0.39366564]
        tensor([    4,    19,    22,  ..., 59973, 59990, 59992]) [0.40149889 0.52985713 0.06864399]
        '''
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            # 这里i从0~n_clients，表示将tensor([    1,    21,    34,  ..., 59952, 59972, 59987])... tensor([    4,    19,    22,  ..., 59973, 59990, 59992]),按照对应的比例分割给n_clients个客户端。
            client_idcs[i] += [idcs]
            # 这里client_idcs是包含n_clients个子列表的列表，每个列表表示对应客户端拥有数据的index。
            '''
            client_idcs: [[tensor([    1,    21,    34,  ..., 24389, 24393, 24399]), tensor([    3,     6,     8,  ..., 16468, 16475, 16483]), ...  tensor([    5,    16,    25,  ..., 24865, 24876, 24881])],
                          [tensor([34025, 34036, 34037,  ..., 59952, 59972, 59987]), tensor([38466, 38480, 38482,  ..., 59979, 59984, 59994]),... tensor([33698, 33707, 33716,  ..., 59983, 59985, 59991])],
                          [tensor([20233, 20245, 20253,  ..., 33619, 33629, 33689]), tensor([38466, 38480, 38482,  ..., 59979, 59984, 59994]),... tensor([33698, 33707, 33716,  ..., 59983, 59985, 59991])]
                         ]
            '''

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    '''
    假设idcs是一个包含多个数组的列表,例如 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    那么调用 np.concatenate(idcs) 将会把这些数组按照默认的轴 0 进行连接，得到一个新的数组[1 2 3 4 5 6 7 8 9]
    '''
    # 上面np.concatenate(idcs)操作,会将子列表里的数据连接起来。这里n_clients=3
    # client_idcs: [array([   1,   21,   34, ..., 7256, 7258, 7264], dtype=int64), array([48419, 48421, 48434, ..., 40499, 40507, 40508], dtype=int64), array([49974, 49987, 49989, ..., 59973, 59990, 59992], dtype=int64)]
    return client_idcs
