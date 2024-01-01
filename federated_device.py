import torch
from torch.utils.data import DataLoader
import numpy as np
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 # 计算差并将结果存储到目标
def subtract_(W, W_old):
    dW = {key : torch.zeros_like(value) for key, value in W.items()}
    for name in W:
        dW[name].data = W[name].data.clone()-W_old[name].data.clone()
    return dW

def eval_op(model,loader):
    model.eval()
    samples, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            _, predicted = torch.max(y_hat.data, 1)
            samples += y.shape[0]
            correct += (predicted == y).sum().item()
    ## 返回准确率
    return correct/samples
    
'''***************************客户端******************************************'''
class Client():
    def __init__(self, model, data, client_id, batch_size=64,train_frac=0.8):
        super().__init__()  

        self.Client_net = model.to(device)
        self.Client_data = data
        self.Client_id = client_id
        self.W = {key : torch.zeros_like(value) for key, value in self.Client_net.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.Client_net.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.Client_net.named_parameters()}

        '''**********************************数据加载*******************************************'''
        # 80%的数据为训练集，20%为测试集
        n_train = int(len(data)*train_frac)
        n_eval = len(data) - n_train 

        data_train, data_eval = torch.utils.data.random_split(self.Client_data, [n_train, n_eval])
        self.data_size = len(data_train) #获取客户端数据集的大小
        self.Client_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.Client_eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=True)
    
    # 训练客户端模型并且计算权重更新
    def compute_weight_update(self, epochs):
        self.W_old = {key : value.clone() for key, value in self.W.items()}#保存旧模型参数
        '''****************训练***************************'''
        self.optimizer = torch.optim.SGD(self.Client_net.parameters(), lr=0.01, momentum=0.9)
        self.Client_net.train()  
        for _ in range(epochs):
            for x, y in self.Client_train_loader: 
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(self.Client_net(x), y)
                loss.backward()
                self.optimizer.step()  
        '''****************训练***************************'''
        self.W = {key : value for key, value in self.Client_net.named_parameters()} #获得新模型参数
        self.dW = subtract_(self.W, self.W_old) #获得参数更新

    #定义了一个与服务器同步的方法，该方法将服务器端的权重复制到客户端的权重上。
    def synchronize_with_server(self, server):
        self.W = {key : value.clone() for key, value in server.W.items()}
        for name in self.W:
            self.Client_net.state_dict()[name].copy_(self.W[name].data)  

    def eval(self):
        acc = eval_op(model=self.Client_net,loader=self.Client_eval_loader)
        return acc
      

import random
'''***************************服务器******************************************'''
class Server():
    def __init__(self, model):
        super().__init__()
        self.model_cache = []
        self.Server_net = model.to(device)
        self.W = {key : value for key, value in self.Server_net.named_parameters()}

    #定义了一个选择客户端的方法，该方法从客户端列表中随机选择一部分客户端，并返回选择的客户端列表。
    def select_clients(self, clients, frac):
        num_clients = int(len(clients) * frac)
        return clients[:num_clients]
        #return random.sample(clients, int(len(clients)*frac)) 
    
    #定义了一个聚合权重更新的方法，该方法将客户端的权重更新聚合到服务器端的权重上，从而得到新的全局模型。
    def aggregate_weight_updates(self, clients):
        all_client_data_size = 0.0
        for client in clients:
            all_client_data_size += client.data_size
        data_rate = {client.Client_id : client.data_size/all_client_data_size for client in clients}
        for name in self.W:
            tmp = torch.sum(torch.stack([client.dW[name].data * data_rate[client.Client_id] for client in clients]), dim=0).clone()
            self.W[name].data += tmp




  


