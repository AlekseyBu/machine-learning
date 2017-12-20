import numpy as np
from scipy.spatial import distance
import pylab as plt

h = 0.5
eps = 1e-5


# Gauss kernel
def kernel(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z ** 2)

# Kvartich kernel
def kernel_gamma(z):
    if (abs(z) <= 1):
        return (1-z**2)**2
    else:
        return 0


def stability(arr):
    for el in arr:
        if el > eps:
            return True
    return False


def nadaray_watson(x, y):
    n = len(x)
    w = []
    for t in range(n):
        w.append([])
        for i in range(n):
            w[t].append(kernel(distance.euclidean(x[t], x[i]) / h))
    w = np.array(w)
    yest = (w * y[:, None]).sum(axis=0) / w.sum(axis=0)
    return yest

def lowess(x, y):
    n = len(x)

    gamma = np.ones(n)
    gamma_old = np.zeros(n)
    yest = np.zeros(n)
    cnt = 0

    while stability(np.abs(gamma - gamma_old)):
        cnt += 1
        w = []
        for t in range(n):
            w.append([])
            for i in range(n):
                w[t].append(kernel(distance.euclidean(x[t], x[i]) / h)*gamma[t])
        w = np.array(w)
        yest = (w * y[:, None]).sum(axis=0) / w.sum(axis=0)

        err = np.abs(yest - y)
        gamma = [kernel_gamma(err[j]) for j in range(n)]
        if (cnt > 5):
            break
    return yest

def generate_wave_set(n_support=1000, n_train=250):
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data


if __name__ == '__main__':
    data = generate_wave_set(100, 80)
    x = data['x_train']
    y = data['y_train']

    # x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
    # x = np.array(x)
    # y = [1, 2, 4.3, 3, 2, 2, 1.5, 1.3, 1.5, 1.7, 1.8, 2]
    # y = np.array(y)

    print("Nadaray Watson's yest:")
    yest_nadaray = nadaray_watson(x, y)
    print(yest_nadaray)
    print("Lowess yest:")
    yest_lowess = lowess(x, y)
    print(yest_lowess)

    plt.clf()
    plt.scatter(x, y, label='data', color="black")
    plt.plot(x, yest_nadaray, label='y nadaray-watson', color="red")
    plt.plot(x, yest_lowess, label='y lowess', color="blue")
    plt.title('Nadaray Watson vs Lowess')
    plt.legend()
    plt.show()
