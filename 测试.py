import numpy as np
import torchvision

# 标签one-hot处理
def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result
 
# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
# sigmoid的一阶导数
def Dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN(object):
    def __init__(self, l0, l1, l2, batch_size=6):
        self.lr = 0.01                                   # 学习率
        self.batch_size = batch_size
        self.lambd = 0.6
        self.W1 = np.random.randn(l0, l1) * 0.01             # 初始化
        self.b1 = np.random.randn(l1) * 0.01
        self.W2 = np.random.randn(l1, l2) * 0.01
        self.b2 = np.random.randn(l2) * 0.01

    # 前向传播
    def train_forward(self, X, y):
        self.X = X                                           # m x 784
        self.z1 = np.dot(X, self.W1) + self.b1               # m x 200, 等于中间层层数
        self.a1 = sigmoid(self.z1)                           # m x 200
 
        self.z2 = np.dot(self.a1, self.W2) + self.b2         # m x 10
        self.a2 = sigmoid(self.z2)                           # m x 10
 
        loss = np.sum(-y*(np.log(self.a2))-(1-y)*(np.log(1-self.a2)))/ 6 + self.lambd*np.sum(self.W1**2)+self.lambd*np.sum(self.W2**2)

        self.d2 = (-y*(1/self.a2)-(1-y)*(-1/(1-self.a2))) * Dsigmoid(self.z2)          #   用于反向传播
        
        return loss, self.a2
    
    def test_forward(self, X, y):
        self.X = X                                           # m x 784
        self.z1 = np.dot(X, self.W1) + self.b1               # m x 200, 等于中间层层数
        self.a1 = sigmoid(self.z1)                           # m x 200
 
        self.z2 = np.dot(self.a1, self.W2) + self.b2         # m x 10
        self.a2 = sigmoid(self.z2)                           # m x 10
 
        loss = np.sum(-y*(np.log(self.a2))-(1-y)*(np.log(1-self.a2)))/ 10000 + self.lambd*np.sum(self.W1**2)+self.lambd*np.sum(self.W2**2)
 
        self.d2 = (-y*(1/self.a2)-(1-y)*(-1/(1-self.a2))) * Dsigmoid(self.z2)          #   用于反向传播

        return loss, self.a2

    # 反向传播
    def backward(self):
        dW2 = np.dot(self.a1.T, self.d2) / self.batch_size
        db2 = np.sum(self.d2, axis=0) / self.batch_size
 
        d1 = np.dot(self.d2, self.W2.T) * Dsigmoid(self.z1)   #  用于反向传播
        dW1 = np.dot(self.X.T, d1) / self.batch_size                       # 784x 200
        db1 = np.sum(d1, axis=0) / self.batch_size                         # 200
 
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

#训练
def train():
    nn = NN(784, 200, 10)
    train_loss_history = []
    for epoch in range(10):
        epoch_loss = 0
        for i in range(0, 60000, nn.batch_size):
            X = train_data.data[i:i+nn.batch_size]
            Y = train_data.targets[i:i+nn.batch_size]
            loss, _ = nn.train_forward(X, Y)
            epoch_loss += loss
            print("epoch:", epoch, "-", i, ":", "{:.3f}".format(loss) )
            nn.backward()
        train_loss_history.append(epoch_loss/10000)
        np.savez("data.npz", w1=nn.W1, b1=nn.b1, w2=nn.W2, b2=nn.b2)
    return train_loss_history

def test():
    r = np.load("data.npz")
    nn = NN(784, 200, 10)
    nn.W1 = r["w1"]
    nn.b1 = r["b1"]
    nn.W2 = r["w2"]
    nn.b2 = r["b2"]
    
    test_loss, result = nn.test_forward(test_data.data, test_data.targets2)
    result = np.argmax(result, axis=1)
    precision = np.sum(result==test_data.targets) / 10000
    print("Precision:", precision)
    return test_loss,precision
 
if __name__ == '__main__':
 
    # Mnist手写数字集
    train_data = torchvision.datasets.MNIST(root='data/', train=True, download=True)
    test_data = torchvision.datasets.MNIST(root='data/', train=False)
    train_data.data = train_data.data.numpy()         # [60000,28,28]
    train_data.targets = train_data.targets.numpy()   # [60000]
    test_data.data = test_data.data.numpy()           # [10000,28,28]
    test_data.targets = test_data.targets.numpy()     # [10000]
 
    # 输入向量处理
    train_data.data = train_data.data.reshape(60000, 28 * 28) / 255.  # (60000, 784)
 
    test_data.data = test_data.data.reshape(10000, 28 * 28) / 255.
 
    # 标签one-hot处理
    train_data.targets = onehot(train_data.targets, 60000) # (60000, 10)
    test_data.targets2 = onehot(test_data.targets, 10000)  # 用于前向传播
 
    train_loss_history = train()
    test_loss_history, test_accuracy= test()

#可视化
test_loss_history2 = np.repeat(test_loss_history,10)
test_loss_history2 = test_loss_history2.tolist()

import matplotlib.pyplot as plt

# summarize history for loss function
plt.plot(train_loss_history)
plt.plot(test_loss_history2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/szya/zl/loss.png')
plt.show()

test_accuracy2 = np.repeat(test_accuracy,10)
test_accuracy2 = test_accuracy2.tolist()
# summarize history for accuracy
plt.plot(test_accuracy2)
plt.title('test accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('/home/szya/zl/accuracy.png')
plt.show()
