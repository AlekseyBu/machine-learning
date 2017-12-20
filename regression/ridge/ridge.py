import numpy as np
import matplotlib.pyplot as plt


def ridge(X, y, cov_matrix, n, alpha = 0.1):
    cov_matrix_ridge = cov_matrix + alpha * np.eye(n)
    print(cov_matrix_ridge)
    eigenval_ridge, eigenvect_ridge = np.linalg.eigh(cov_matrix_ridge)
    print("новые собственные значения:")
    print(", ".join("%.2f" % f for f in eigenval_ridge))
    return np.linalg.inv(cov_matrix_ridge).dot(X.T).dot(y)


def generate_wave_set(n_support=1000, n_train=250):
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data


if __name__ == '__main__':
    n = 80
    data = generate_wave_set(100, n)
    X = data['x_train']
    y = data['y_train']

    plt.scatter(X, y, label='data', color="black")
    
    fX = np.array([X]).T
    fX = np.concatenate((fX, np.power(fX, 2), np.power(fX, 3), 2*fX, 3*fX), axis=1)

    cov_matrix_standart = fX.T.dot(fX)
    print(cov_matrix_standart)
    # get the eigenvalues and eigenvectors
    eigenval, eigenvect = np.linalg.eigh(cov_matrix_standart)
    print("первоначальные собственные значения:")
    print(", ".join("%.2f" % f for f in eigenval))

    ans = ridge(fX, y, cov_matrix_standart, n=5, alpha=0.3)

    print(ans)

    plt.legend()
    plt.show()