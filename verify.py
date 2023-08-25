from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 转换数据类型
X = np.genfromtxt(r"train_dataset_X_scaled.txt", delimiter=',', dtype=np.float64)
y = np.genfromtxt(r"train_dataset_y_orthogonal.txt", delimiter=',', dtype=np.float64)

# 建模
from neuralNetwork import NeuralNetwork
nn = NeuralNetwork([X.shape[1], 100, y.shape[1]], 'logistic')

k = 5  # 设置折数
kf = KFold(n_splits=k, shuffle=True)

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 在训练集上训练模型
    nn.fit(X_train, y_train)

    # 在验证集上进行预测
    y_pred = nn.predict(X_val)
    y_pred = np.array(y_pred)
    y_pred = np.column_stack([1 - y_pred, y_pred])

    
    # 使用阈值处理
    y_pred_binary = np.where(y_pred >= 0.5, 1, 0)

    # 转换预测结果为单标签形式
    y_pred_binary = np.argmax(y_pred, axis=1)
    y_val_true = np.argmax(y_val, axis=1)

    # 计算准确率和F1分数
    accuracy = accuracy_score(y_val_true, y_pred_binary)
    f1 = f1_score(y_val_true, y_pred_binary, average='macro')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
