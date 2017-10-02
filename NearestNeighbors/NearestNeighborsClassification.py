import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score

Kvalues = list(range(1,25))
#odd = list(filter(lambda x: x % 2 != 0, myList))

wine = datasets.load_wine()

#take the first two features
X = wine.data[:, :2]
y = wine.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#coordinates
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for algorithm in ['brute', 'ball_tree', 'kd_tree']:
    for weights in ['uniform', 'distance']:
        cv_scores = []
        for k in Kvalues:
            clf = neighbors.KNeighborsClassifier(n_neighbors = k,
                                                 weights=weights,
                                                 algorithm=algorithm)
            clf.fit(X, y)
            scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())
        # changing to misclassification error
        MSE = [1 - x for x in cv_scores]
        # determining best k
        optimal_k = Kvalues[MSE.index(min(MSE))]
        print("The optimal number of neighbors for algorithm = '%s' and weights = '%s' is %d" 
              % (algorithm, weights, optimal_k))

#         # plot misclassification error vs k
#         plt.plot(Kvalues, MSE)
#         plt.xlabel('Number of Neighbors K')
#         plt.ylabel('Misclassification Error')
      
        kNN = neighbors.KNeighborsClassifier(n_neighbors = optimal_k, 
                                             weights=weights,
                                             algorithm=algorithm)
        kNN.fit(X, y)
        Z = kNN.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s', algorithm = '%s')"
                  % (optimal_k, weights, algorithm))

plt.show()