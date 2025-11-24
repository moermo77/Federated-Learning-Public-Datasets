import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json

def truncnorm(mu,sigma,num,lower,upper):
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)#有区间限制的随机数
    return X.rvs(num)

def func(x):
    #return x*x-3*x+0.1
    return x*x-3*x+2


def plot_data(data, label):
    legend = []
    for i in range(len(data)):
        print("client"+str(i)+":",np.mean(data[i]),np.var(label[i]))
        plt.scatter(data[i],label[i],s=2)
        legend.append("client"+str(i))
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel("y")
    #plt.savefig(os.path.join(path,'main.png'))
    plt.show()


def data_generator(client_num, clinet_data_size, func, noise, mean, sigma, share = False, noise_rate = 0.4, shared_data_size = 10, data_lower_bound=-2, data_upper_bound=5, data_split=0.85):
    '''
    生成实验数据
    :param clinet_num: int, 参与联邦学习client节点数
    :param clinet_data_size: int, 单个节点数据量
    :param func: function, 需要拟合的曲线函数
    :param noise: list, 不同client数据添加的噪声
    :param mean: list, 不同client数据均值
    :param sigma: int, 不同client数据标准差
    :param share: bool, 是否加入共享数据
    :param noise_rate: float, 噪声系数α
    :param shared_data_size: int, 每个client分到的共享数据的规模
    :param data_lower_bound: float, client数据分布下限
    :param data_pper_bound: float, client数据分布下上限
    :param data_split: float, 训练集划分比例
    :return:
    '''
    assert data_lower_bound < data_upper_bound
    assert clinet_data_size*data_split-1> shared_data_size
    assert len(mean) == len(sigma) == client_num
    np.random.seed(0)
    #测试集训练集划分
    train_size = int(data_split*clinet_data_size)
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    #共享数据
    data_share = np.linspace(data_lower_bound,data_upper_bound,shared_data_size*client_num)
    label_share = func(data_share)
    label_noise_share = label_share + noise[0][0:shared_data_size*client_num]*noise_rate
    for i in range(client_num):
        x = truncnorm(mean[i],sigma[i],clinet_data_size,data_lower_bound,data_upper_bound)
        y = func(x)+noise[i]*noise_rate
        #训练集划分
        data,label = x[0:train_size],y[0:train_size]
        train_data.append(data)
        train_label.append(label)
        #测试集划分
        data,label = x[train_size:clinet_data_size],y[train_size:clinet_data_size]
        #若使用共享数据
        if share:
            data[0:shared_data_size] = data_share[i:shared_data_size*client_num:client_num]
            label[0:shared_data_size] = label_noise_share[i:shared_data_size*client_num:client_num]

        test_data.append(data)
        test_label.append(label)

    #可视化数据
    plot_data(train_data, train_label)

    #导出数据
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    test_data = np.concatenate(test_data)
    test_label = np.concatenate(test_label)
    np.savetxt("train_data.txt",train_data.reshape(-1,1))
    np.savetxt("train_label.txt",train_label.reshape(-1,1))
    np.savetxt("test_data.txt",test_data.reshape(-1,1))
    np.savetxt("test_label.txt",test_label.reshape(-1,1))

    #导出数据划分
    train_idx = {}
    test_idx = {}
    test_size = clinet_data_size-train_size
    for i in range(client_num):
        train_idx[i] = np.linspace(i*train_size,(i+1)*train_size-1,train_size).astype(np.int64).tolist()
        test_idx[i] = np.linspace(i*test_size,(i+1)*test_size-1,test_size).astype(np.int64).tolist()
    train_idx_str = json.dumps(train_idx)
    test_idx_str = json.dumps(test_idx)
    a = open("train_idx.txt",'w')
    a.write(train_idx_str)
    a.close()
    a = open("test_idx.txt",'w')
    a.write(test_idx_str)
    a.close()




if __name__ == "__main__":
    np.random.seed(0)
    client_data_size = 200
    client_num = 5
    noise = []
    noise_0 = np.random.normal(0,0.2,client_data_size)#正态分布
    mean = [-0.5,0.5,1.5,2.5,3.5]
    sigma = [0.2, 0.2, 0.2,0.2,0.2]
    for x in range(client_num):
        noise.append(noise_0)
    data_generator(client_num, client_data_size, func, noise, mean, sigma, False, noise_rate = 0.4, shared_data_size = 10, data_lower_bound=-2, data_upper_bound=5,data_split=0.85)



