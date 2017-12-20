import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def func(val, x):
	c0, c1, c2 = val
	return c0*np.exp(c1*x) + c2


def derivative(val, x, y):
    c0, c1, c2 = val
    dc0 = (c0*np.exp(c1*x) + c2 - y)*np.exp(c1*x)
    dc1 = (c0*np.exp(c1*x) + c2 - y)*c0*x*np.exp(c1*x)
    dc2 = (c0*np.exp(c1*x) + c2 - y)
    return np.array([dc0, dc1, dc2])


def SSE(val, X, y):
    res = 0
    for i in range(len(X)):
        res += 0.5*(func(val, X[i]) - y[i])**2
    return res


def stability(arr, eps):
    for el in arr:
        if el > eps:
            return True
    return False


def gradientDescent(X, y, gamma = 0.01, maxcnt = 1000):
    params_cnt = 3
    np.random.seed(363445)
    params = np.random.randn(params_cnt) * 1.5
    params = np.abs(params)
    while True:
        grad = np.zeros(params_cnt)
        for i in range(len(X)):
            grad += derivative(params, X[i], y[i])
        norm = np.linalg.norm(grad)
        grad /= norm
        params -= grad * gamma
        # print("SSE = %.3f" % (SSE(params, X, y)))
        maxcnt = maxcnt - 1
        if (maxcnt == 0):
            break
    return params

def generate_set(n = 100):
    data = {}
    data['x'] = np.linspace(1, 10, num=n)
    data['y'] = np.exp(data['x']) + 2500*np.random.rand(data['x'].size)
    return data

if __name__ == '__main__':
    n = 50
    data = generate_set(n)
    X = data['x']
    y = data['y']

    params_vec = gradientDescent(X, y, gamma=0.001, maxcnt=1000)
    print(params_vec)
    yest = [func(params_vec, X[i]) for i in range(n)]

    plt.scatter(X, y, label='data', color="black")
    plt.plot(X, yest, label='y prediction', color="blue")
    plt.title('Nonlinear regression')
    plt.legend()
    plt.show()