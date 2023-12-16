import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

#用于记录实验结果的简单日志类。它具有一个log方法，
class ExperimentLogger:
    #用于将键值对记录到实例的属性中。如果键不存在，则创建一个新的属性并将值存储为列表；如果键已经存在，则将值添加到相应的列表中。
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]


def show_acc(client_acc_stats, communication_rounds):
    clear_output(wait=True)
    plt.figure(figsize=(8,4))

    acc_mean = np.mean(client_acc_stats.clients_acc, axis=1)
    acc_std = np.std(client_acc_stats.clients_acc, axis=1)
    plt.fill_between(client_acc_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(client_acc_stats.rounds, acc_mean, color="C0")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.title("client_acc")
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)

    plt.show()