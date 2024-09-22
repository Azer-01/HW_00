import math
import matplotlib.pyplot as plt
import time
#矩阵乘法
def dot(X, theta):
    if len(X[0]) != len(theta):
        raise ValueError("矩阵相乘出错")
    result = [0] * len(X)
    for i in range(len(X)):
        result[i] = 0
        for j in range(len(theta)):
            result[i] += X[i][j] * theta[j]
    return result

#矩阵转置
def transpose(X):
    return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

#梯度计算
def grad_cou(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    errors = [predictions[i] - y[i] for i in range(m)]
    gradients = [0] * len(theta)
    X_transpose = transpose(X)

    for j in range(len(theta)):
        for i in range(m):
            gradients[j] += (1 / m) * X_transpose[j][i] * errors[i]

    return gradients

#多项式特征生成
def poly_features(x, degree):
    return [[x_i ** i for i in range(degree + 1)] for x_i in x]  #生成多项式特征

#预测函数
def predict(X, theta):
    return dot(X, theta)

#损失函数
def loss(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    return (1 / (2 * m)) * sum((predictions[i] - y[i]) ** 2 for i in range(m))

#梯度下降
def grad(X, y, theta, alpha, num_iters):
    m = len(y)
    loss_history = []

    for i in range(num_iters):
        gradients = grad_cou(X, y, theta)
        for j in range(len(theta)):
            theta[j] = theta[j] - alpha * gradients[j]
        #记录损失
        loss_history.append(loss(X, y, theta))
    return theta, loss_history

def show(x, y, y_pred):
    plt.plot(x, y, 'r-', label='sin(x)')  # 目标函数 sin(x)
    plt.plot(x, y_pred, 'b-', label='Polynomial Fit')  # 多项式拟合结果
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('5th Degree Polynomial Fit and sin(x)')
    plt.savefig('sinx.png')
    plt.show()

def main():
    #初始化
    alpha = 0.0002  #学习率
    iters = 50000  #迭代次数
    x = [i * 6 / 99 - 3 for i in range(100)]  # 在[-3, 3]区间上生成100个点
    y = [math.sin(x_i) for x_i in x]
    X_poly = poly_features(x, 5)
    theta = [0] * len(X_poly[0])
    #模型训练
    theta, loss_history = grad(X_poly, y, theta, alpha, iters)
    #预测
    y_pred = predict(X_poly, theta)
    #展示结果
    show(x, y, y_pred)
if __name__ == "__main__":
    main()
