import matplotlib.pyplot as plt
#数据集
X = [[700], [800], [850], [900], [950], [1000], [1050], [1100], [1150], [1200],
     [1250], [1300], [1350], [1400], [1450], [1500], [1550], [1600], [1650], [1700]]

#目标房价
y = [240000, 250000, 260000, 270000, 280000, 290000, 300000, 310000, 320000, 330000,
     340000, 350000, 360000, 370000, 380000, 390000, 400000, 410000, 420000, 430000]

#矩阵乘法
def dot(X, theta):
    if len(X[0]) != len(theta):
        print("矩阵相乘出错")
    result = [0] * len(X)
    for i in range(len(X)):
        result[i] = 0
        for j in range(len(theta)):
            result[i] += X[i][j] * theta[j]
    return result

#归一化数据
def normalize(X):
    mean = sum([row[0] for row in X]) / len(X)
    std = (sum([(row[0] - mean) ** 2 for row in X]) / len(X)) ** 0.5
    return [[(row[0] - mean) / std] for row in X]

def add_bias(X):
    return [[1] + row for row in X]

#线性回归预测
def predict(X, theta):
    return dot(X, theta)

#损失函数
def loss(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    return (1 / (2 * m)) * sum([(predictions[i] - y[i]) ** 2 for i in range(m)])

#梯度下降
def grad(X, y, theta, alpha, num_iters):
    m = len(y)
    loss_history = []

    for i in range(num_iters):
        predictions = predict(X, theta)
        errors = [predictions[i] - y[i] for i in range(m)]
        #计算梯度并更新参数
        gradients = [0] * len(theta)
        for j in range(len(theta)):
            gradients[j] = (1 / m) * sum([errors[i] * X[i][j] for i in range(m)])
        theta = [theta[k] - alpha * gradients[k] for k in range(len(theta))]

        loss_history.append(loss(X, y, theta))
    return theta, loss_history

#对比可视化
def show(predictions):
    plt.plot(y, 'ro', label='Actual Price')
    plt.plot(predictions, 'b-', label='Predicted Price')
    plt.xlabel('Sample')
    plt.ylabel('House Price')
    plt.legend()
    plt.title('Prediction Price and Actual Price')
    plt.show()

def main():
    #标准化处理
    X_norm = normalize(X)
    X_bias = add_bias(X_norm)
    #初始化
    theta = [0] * len(X_bias[0])
    alpha = 0.01  #学习率
    num_iters = 1000  #迭代次数

    theta, loss_history = grad(X_bias, y, theta, alpha, num_iters)#模型

    predictions = predict(X_bias, theta)#预测值
    show(predictions)
if __name__=="__main__":
    main()
