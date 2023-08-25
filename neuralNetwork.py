import numpy as np
class NeuralNetwork:
    # 生长函数及其倒数
    def logistic(self,x):
        return 1/(1 + np.exp(-x))
    def logistic_derivative(self,x):
        return self.logistic(x)*(1-self.logistic(x))
    #定义双曲函数及其导数
    def tanh(self,x):
        return np.tanh(x)
    def tanh_deriv(self,x):
        return 1.0 - np.tanh(x)**2
    
    #初始化，layes表示的是一个list，如[10,5,2]表示第一层10个神经元
    #第二层5个神经元，第三层2个神经元。至少2层
    #权重的层数是神经元层数-1
    #权重的初值的赋值，因为不知道，通常用随机数来产生，产生在【-0.5，0.5】之间的，
    #一般需要正负概率相等。
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = self.logistic
            self.activation_deriv = self.logistic_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_deriv = self.tanh_deriv
        self.weights = []   #权重的list
        #从第二层开始，前后节点连接权重的初始化。因为第0层为输入，最后层输出
        for i in range(len(layers) - 1):
            #对当前节点的前趋赋值,为使权重有正有负号，均衡一点，所以采用下面公式
            #分布在-1到+1之间
            #np.random.random((n,m)),产生n*m矩阵，每个数是0-1随机小数，
            # *2后，成为0-2之间，再-1，变成-1到+1之间。
            if i==0: # 输入层，加偏置
                w=2*np.random.random((layers[i]+1 , layers[i + 1]))-1
                self.weights.append(w*0.5)
            else:
                w=2*np.random.random((layers[i] , layers[i + 1]))-1
                self.weights.append(w*0.5)
                
    
    #训练函数，X矩阵，每行是一个样本的特征 ，y是其对应的分类，learning_rate 学习速率，
    # epochs，神经网络进行学习的最大次数
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        #X = np.atleast_2d(X) #确定X至少是二维的数据
        temp = np.ones(X.shape[0]) #初始化矩阵，+1列为bias
       
        X = np.c_[X,temp]
        for k in range(epochs):
            #随机选取一行，对神经网络进行更新
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            #完成所有正向的更新
            for j in range(len(self.weights)):
                a.append(self.activation(np.dot(a[j], self.weights[j])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            #开始反向误差传播，更新权重
            for j in range(len(a) - 2, 0, -1): # 从倒数第2层到开始层
                deltas.append(deltas[-1].dot(self.weights[j].T)*self.activation_deriv(a[j]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

                #预测函数
    def predict(self, x):
        temp = np.ones(x.shape[0])
        x=np.c_[x,temp]
        ans=[]
        for a in x:
            for w in self.weights:
                a = self.activation(np.dot(a, w))
            ans.append(np.argmax(a))
        return ans
                #预测函数
    '''
    def predict1(self, x):
        temp = np.ones(x.shape[0])
        x=np.c_[x,temp]
        ans=[]
        for a in x:
            for w in self.weights:
                a = self.activation(np.dot(a, w))
            #ans.append(a[0])
            ans.append(np.argmax(a))  #对分类问题，用此语句

        return ans