import numpy as np
import matplotlib.pyplot as plt

# 生成原始数据集
def generate_nonlinear_dataset(n_samples):
    np.random.seed(0)
    class_1 = np.random.normal(loc=[-1, 1], scale=0.5, size=(n_samples, 2))
    class_2 = np.random.normal(loc=[1, -1], scale=0.5, size=(n_samples, 2))
    data = np.concatenate((class_1, class_2))
    labels = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))
    return data, labels

# 分割数据集
def split_dataset(data, labels, train_ratio, test_ratio):
    assert train_ratio + test_ratio == 1, "划分比例之和必须等于1"
    n_samples = data.shape[0]
    n_train = int(n_samples * train_ratio)
    n_test = n_samples - n_train
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    X_train, X_test = data[train_indices], data[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    return X_train, X_test, y_train, y_test

# 预处理训练集数据
def preprocess_data(X,Xtest):
    # 特征缩放
    aver=X.mean(axis=0)
    X -= aver
    X_temp = np.abs(X)
    col_max = X_temp.max(axis=0)
    X /= col_max
    Xtest -= aver
    Xtest /= col_max
    return X,Xtest

# 正交化标签
def orthogonalize_labels(y):
    labelstrain = []
    for label in y:
        one = 2 * [0]
        if label == 0:
            one[0] = 1
        else:
            one[1] = 1
        labelstrain.append(one)
    return np.array(labelstrain)

# 生成原始数据集
data, labels = generate_nonlinear_dataset(1000)

# 分割为训练集和测试集
X_train, X_test, y_train, y_test = split_dataset(data, labels, train_ratio=0.9, test_ratio=0.1)

# 预处理数据
X_train_scaled, X_test_scaled = preprocess_data(X_train,X_test)


# 正交化标签
y_train_orthogonal = orthogonalize_labels(y_train)

# 保存为txt文件
np.savetxt('train_dataset_raw.txt', np.column_stack((X_train, y_train)), delimiter=',')
# 导出训练集数据
np.savetxt('train_dataset_X_scaled.txt', X_train_scaled.astype(float), delimiter=',')
np.savetxt('train_dataset_y_orthogonal.txt', y_train_orthogonal.astype(float), delimiter=',')
# 导出测试集数据
np.savetxt('test_dataset_X_scaled.txt', X_test_scaled, delimiter=',')
np.savetxt('test_dataset_y.txt', y_test, delimiter=',')




# 可视化数据
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 绘制原始训练集
axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.7)
axs[0, 0].set_title('Raw Training Dataset')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')

# 绘制缩放后的训练集
axs[0, 1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='bwr', alpha=0.7)
axs[0, 1].set_title('Scaled Training Dataset')
axs[0, 1].set_xlabel('X_scaled')
axs[0, 1].set_ylabel('Y_scaled')

# 绘制原始测试集
axs[0, 2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.7)
axs[0, 2].set_title('Raw Test Dataset')
axs[0, 2].set_xlabel('X')
axs[0, 2].set_ylabel('Y')

# 绘制缩放后的测试集
axs[1, 0].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='bwr', alpha=0.7)
axs[1, 0].set_title('Scaled Test Dataset')
axs[1, 0].set_xlabel('X_scaled')
axs[1, 0].set_ylabel('Y_scaled')

# 隐藏多余的子图
axs[1, 1].axis('off')
axs[1, 2].axis('off')

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig('dataset_visualization.png')

# 显示图像
plt.show()

# 建模数据导入
# X=np.loadtxt(r"train_dataset_X_scaled.txt",dtype=str)
# y=np.loadtxt(r"train_dataset_y_orthogonal.txt",dtype=str)

# 转换数据类型
X = np.genfromtxt(r"train_dataset_X_scaled.txt", delimiter=',', dtype=np.float64)
y = np.genfromtxt(r"train_dataset_y_orthogonal.txt", delimiter=',', dtype=np.float64)


# 建模
from neuralNetwork import NeuralNetwork
nn =NeuralNetwork([X.shape[1],100,y.shape[1]],'logistic')
nn.fit(X,y,epochs=10000)

# 预测及分类报告
Xtest=np.genfromtxt(r"test_dataset_X_scaled.txt", delimiter=',', dtype=np.float64)
ytest=np.genfromtxt(r"test_dataset_y.txt", delimiter=',', dtype=np.float64)

predictions=nn.predict(Xtest)
print(predictions)
print(ytest)
from sklearn.metrics import classification_report
print (classification_report(ytest, predictions))


