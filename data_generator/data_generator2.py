import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import os
import  random
def truncnorm(mu,sigma,num,lower,upper):
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)#有区间限制的随机数
    return X.rvs(num)

def func(x):
    return x*x-3*x+0.1
def func_2(x):
    y_1 = np.pi/2*(x**2+4*x+3)*(x<-1)
    y_2 = np.sin(np.pi*(x+1))*(x>-1)*(x<=1)
    y_3 = np.pi*np.log(x)*(x>1)
    y_1[np.where(np.isnan(y_1))] = 0
    y_2[np.where(np.isnan(y_2))] = 0
    y_3[np.where(np.isnan(y_3))] = 0
    return y_1+y_2+y_3
def func_3(x):
    return x[:,0]**2+x[:,1]**2
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


def data_generator(client_num, clinet_data_size, func, noise, mean, sigma, share = True, \
                   shared_data_size = 10, data_lower_bound=-2, data_upper_bound=5, \
                   data_split=0.85,train_dir = None,test_dir = None):
    '''
    生成实验数据
    :param clinet_num: int, 参与联邦学习client节点数
    :param clinet_data_size: int, 单个节点数据量
    :param func: function, 需要拟合的曲线函数
    :param noise: list, 不同client数据添加的噪声
    :param mean: list, 不同client数据均值
    :param sigma: int, 不同client数据标准差
    :param share: bool, 是否加入共享数据
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
    shared = []
    for i in range(client_num):
        x = truncnorm(mean[i],sigma[i],clinet_data_size,data_lower_bound,data_upper_bound)
        #x = truncnorm(mean[i],sigma[i],[clinet_data_size,2],data_lower_bound,data_upper_bound)
        y = func(x)+0.2*noise[i]
        #y = (func_3(x)> 9).astype(np.int)
        if share:
            #训练集划分
            data,label = x[0:train_size],y[0:train_size]
            index = np.random.choice(data.shape[0],shared_data_size,replace=False#从data中随机采样
            shared.append(np.concatenate([np.expand_dims(data[index],1), \
                                          np.expand_dims(label[index],1)],1))
            data = np.delete(data,index)
            label = np.delete(label,index)
            train_data.append(data)
            train_label.append(label)
            #测试集划分
            data,label = x[train_size:],y[train_size:]
            test_data.append(data)
            test_label.append(label)
        else:
            #训练集划分
            data,label = x[0:train_size],y[0:train_size]
            train_data.append(data)
            train_label.append(label)
            #测试集划分
            data,label = x[train_size:],y[train_size:]
            test_data.append(data)
            test_label.append(label)

        #若使用共享数据
        '''
        if share:
            data[0:shared_data_size] = data_share[i:shared_data_size*client_num:client_num]
            label[0:shared_data_size] = label_noise_share[i:shared_data_size*client_num:client_num]
        '''

    if share:
        shared = np.concatenate(shared,0)
        for i in range(client_num):
            index = np.random.choice(shared.shape[0],shared_data_size,replace=False)
            train_data[i] = np.append(train_data[i], shared[:,0])
            train_label[i] = np.append(train_label[i], shared[:,1])
            #shared = np.delete(shared,index,0)
                #可视化数据
    plot_data(train_data, train_label)

    #导出数据
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    test_data = np.concatenate(test_data)
    test_label = np.concatenate(test_label)
    np.savetxt(os.path.join(train_dir,"train_data.txt"),train_data.reshape(-1,1))
    np.savetxt(os.path.join(train_dir,"train_label.txt"),train_label.reshape(-1,1))
    np.savetxt(os.path.join(test_dir,"test_data.txt"),test_data.reshape(-1,1))
    np.savetxt(os.path.join(test_dir,"test_label.txt"),test_label.reshape(-1,1))

    #导出数据划分
    train_idx = {}
    test_idx = {}
    test_size = clinet_data_size-train_size
    for i in range(client_num):
        train_idx[i] = np.linspace(i*train_size,(i+1)*train_size-1,train_size).astype(np.int).tolist()
        test_idx[i] = np.linspace(i*test_size,(i+1)*test_size-1,test_size).astype(np.int).tolist()
    train_idx_str = json.dumps(train_idx)
    test_idx_str = json.dumps(test_idx)
    a = open(os.path.join(train_dir,"train_idx.txt"),'w')
    a.write(train_idx_str)
    a.close()
    a = open(os.path.join(test_dir,"test_idx.txt"),'w')
    a.write(test_idx_str)
    a.close()




if __name__ == "__main__":
    np.random.seed(0)
    #参数
    client_data_size = 200
    client_num = 5
    name = 'Experiment共享4\\不加'
    if_share = True
    share_rate = 0.05

    path = os.getcwd()[:-15]
    train_dir = os.path.join(path,'data',name,'train')
    test_dir = os.path.join(path,'data',name,'test')
    model_dir = os.path.join(path,'data',name,'model')
    for folder in [train_dir,test_dir,model_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    noise_rate = 0
    noise = []
    mean_noise = np.array([0,0,0,0,0])*noise_rate
    sigma_noise = np.array([0.2,0.2,0.2,0.2,0.2])*1
    var = locals()
    for i in range(client_num):
        var['noise_'+str(i)] = np.random.normal(mean_noise[i], sigma_noise[i],client_data_size)
    mean = [-0.5,0.5,1.5,2.5,3.5]
    sigma = np.ones(client_num)*0.2
    for x in range(client_num):
        noise.append(var['noise_'+str(x)])
        #noise.append(var['noise_4'])
    data_generator(client_num, client_data_size, func, noise, mean, \
                   sigma, if_share, shared_data_size = int(share_rate*client_data_size), data_lower_bound=-2, \
                   data_upper_bound=5,data_split=0.85,train_dir=train_dir,test_dir=test_dir)




