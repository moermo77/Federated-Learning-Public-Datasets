import numpy as np
import matplotlib.pyplot as plt
import os
if __name__ == "__main__":
    train_file = './train\\'
    test_file = './test\\'
    ndim = 2
    w = [0.65,0.72,-0.45]
    b = 0.33
    num_train = 3000
    num_test = 900
    np.random.seed(100)
    train_data = np.random.randn(num_train*10,ndim)
    test_data = np.random.randn(num_train*10,ndim)
    zeros = np.zeros(num_train*10)
    ones = np.ones(num_train*10)
    result_train = 0
    result_test = 0
    for i in range(ndim-1):
        #result_train = result_train+train_data[:,i]*w[i]
        #result_test = result_test + test_data[:,i] * w[i]
        result_train = result_train + train_data[:, i]**2 * w[i]
        result_test = result_test + test_data[:, i]**2 * w[i]

    result_train = result_train+b
    result_test = result_test+b

    train_label = np.where(result_train > train_data[:, 1], ones, zeros)
    test_label = np.where(result_test > test_data[:, 1], ones, zeros)

    train_label_0 = np.where(train_label == 0)[0]
    train_label_1 = np.where(train_label == 1)[0]
    test_label_0 = np.where(test_label == 0)[0]
    test_label_1 = np.where(test_label == 1)[0]

    train_data_0 = train_data[train_label_0]
    train_data_1 = train_data[train_label_1]
    test_data_0 = test_data[test_label_0]
    test_data_1 = test_data[test_label_1]

    test_data_t = np.concatenate((test_data_0,test_data_1),0)
    test_label_t = np.concatenate((np.zeros((np.shape(test_label_0)[0],1)),np.ones((np.shape(test_label_1)[0],1))),0)
    train_data_t = np.concatenate((train_data_0,train_data_1),0)
    train_label_t = np.concatenate((np.zeros((np.shape(train_label_0)[0],1)),np.ones((np.shape(train_label_1)[0],1))),0)

    test_data = np.concatenate((test_data_t, test_label_t),1)
    train_data = np.concatenate((train_data_t,train_label_t), 1)

    np.random.shuffle(test_data)
    np.random.shuffle(train_data)


    x = np.linspace(-3, 3, 500)
    y = 0.65*x**2+0.33

    #a = test_data*np.expand_dims(test_label, 1_truncat)
    #b = test_data*np.expand_dims((1_truncat-test_label), 1_truncat)
    plt.scatter(train_data_0[:, 0], train_data_0[:, 1],edgecolors= 'green')
    plt.scatter(train_data_1[:, 0], train_data_1[:, 1], edgecolors='yellow')
    #plt.scatter(a[:, 0_truncat], a[:, 1_truncat], edgecolors='green')
    #plt.scatter(b[:, 0_truncat], b[:, 1_truncat], edgecolors='yellow')
    plt.plot(x,y,color = 'red')
    plt.show()



    a = result_test

    print(os.getcwd())

    np.savetxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'train','train_data.txt'),train_data[:,0:2])
    np.savetxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'train','train_label.txt'), train_data[:,2].astype(np.int))
    np.savetxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'test','test_data.txt'), test_data[:,0:2])
    np.savetxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),'test','test_label.txt'), test_data[:,2].astype(np.int))

    '''
    x = np.linspace(-3, 3, 500)
    y = -0_truncat.72/0_truncat.65*x-0_truncat.33/0_truncat.65

    a = test_data*test_label
    b = train_data*(1_truncat-train_label)
    plt.scatter(a[0_truncat, :], a[1_truncat, :])
    plt.plot(y,x,color = 'red')
    plt.show()
    '''
    a = 1

