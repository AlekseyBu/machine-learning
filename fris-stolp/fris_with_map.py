import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

iris = datasets.load_iris()

# take two features
X = iris.data[:, [2,3]]
y = iris.target

# wine = datasets.load_wine()
#
# # take the first two features
# X = wine.data[:, [6,9]]
# y = wine.target

n = len(X)  # data length
xl = np.arange(n)  # data indexes

eps = 1e-5
alpha = 0.7
tetta = 0.1


# substraction of two sets
def sets_diff(a1, a2):
    a1 = np.atleast_1d(a1)
    a2 = np.atleast_1d(a2)
    if not a1.size:
        return a1
    if not a2.size:
        return a1
    return np.setdiff1d(a1,a2)


# union of two sets
def sets_union(a1, a2):
    a1 = np.atleast_1d(a1)
    a2 = np.atleast_1d(a2)
    if not a1.size:
        return a2
    if not a2.size:
        return a1
    return np.union1d(a1,a2)


# fris-function
def fris(a, b, x):
    return (distance.euclidean(a, x) - distance.euclidean(a, b)) / (distance.euclidean(a, x) + distance.euclidean(a, b) + eps)


# returns nearest to u object from U
def nearest_neighbor(u, U):
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(U)
    return U[nbrs.kneighbors(u, return_distance=False)]


# return new etalon (index) for Y class based on existing etalons 'etalons' and set of Xy elements of Y class
def find_etalon(Xy, etalons):
    length = len(Xy)
    if length == 1:
        return Xy[0]
    etalonsval = []
    for i in etalons:
        etalonsval.append(X[i])
    etalonsval = np.asarray(etalonsval)
    D, T, E = [], [], []
    i = 0
    arrdiff = sets_diff(xl, Xy) # X/Xy
    for x in range(length):
        # defence
        sum = 0
        for u in range(length):
            if u == x:
                continue
            sum = sum + fris(X[Xy[u]], X[Xy[x]], nearest_neighbor(np.reshape(X[Xy[u]], (1,-1)), etalonsval))
        # defence
        D.append(sum/(len(Xy) - 1))

        # tolerance
        sum = 0
        for v in range(len(arrdiff)):
            sum = sum + fris(X[arrdiff[v]], X[Xy[x]], nearest_neighbor(np.reshape(X[arrdiff[v]], (1,-1)), etalonsval))
        # tolerance
        T.append(sum/(len(arrdiff)))

        # efficiency
        E.append(alpha*D[i] + (1-alpha)*T[i])

        i = i + 1
    return Xy[np.argmax((E))] #index from Xy


def fris_stolp():
    xByClass = []
    for i in np.unique(y):
        xByClass.append(np.arange(n)[y == i])

    # step1: finding etalon0
    etalon0 = []
    for i in (np.unique(y)):
        etalon0.append(find_etalon(xByClass[i], sets_diff(xl, xByClass[i])))
    etalon0 = np.asarray(etalon0)
    print("Initial etalons:")
    print(etalon0)

    # step2: initializing etalons for all classes
    etalonsunion = []
    for i in (np.unique(y)):
        etalonsunion = sets_union(etalonsunion, etalon0[i])

    etalons = []
    for i in (np.unique(y)):
        etalons.append(find_etalon(xByClass[i], sets_diff(etalonsunion, etalon0[i])))
    print("etalons:")
    print(etalons)

    # step3: repeat steps4-6 until X is not empty
    etalonslist = [] # use this trick for comfortable array substaction
    for i in (np.unique(y)):
        etalonslist = sets_union(etalonslist, etalons[i])

    xindexes = xl
    while (len(xindexes)):
        # step4: initialize correct obj
        correct = []
        for i in range(len(xindexes)):
            index = xindexes[i]  # element index
            x = X[index]
            yclass = y[index]

            etalonsYVal = []
            for j in np.atleast_1d(etalons[yclass]):
                etalonsYVal.append(X[j])
            etalonsYVal = np.asarray(etalonsYVal)

            etalonsdif = sets_diff(etalonslist, etalons[yclass])
            etalonsval = []
            for j in np.atleast_1d(etalonsdif):
                etalonsval.append(X[j])
            etalonsval = np.asarray(etalonsval)

            val = fris(x, nearest_neighbor(np.reshape(x, (1,-1)), etalonsYVal), nearest_neighbor(np.reshape(x, (1,-1)), etalonsval))
            if (val > tetta):
                correct.append(index)
        print("correct")
        print(correct)
        if (not len(correct)):
            break

        # step5: delete correct from xByClass and xIndexes
        for i in np.unique(y):
            xByClass[i] = sets_diff(xByClass[i], correct)
        xindexes = sets_diff(xindexes, correct)

        # step6: add new etalon for each class
        for i in np.unique(y):
            if (len(xByClass[i])):
                etalons[i] = sets_union(etalons[i], find_etalon(xByClass[i], sets_diff(etalonslist,etalons[i])))

        etalonslist = []
        for i in np.unique(y):
            etalonslist = sets_union(etalonslist, etalons[i])

        print(etalons)

    return etalons

ans = fris_stolp()
print("final etalons:")
print(ans)
etalons = []
for i in np.unique(y):
    etalons = sets_union(etalons, ans[i])
colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(X[:,0], X[:,1], c=y, cmap=colors, s=30)
plt.scatter(X[etalons,0], X[etalons,1], c=y[etalons], cmap=colors, s=300)
# plt.title("Etalons for iris flower data set with alpha = %f" % alpha)
plt.title("Etalons for wine data set with alpha = %f" % alpha)
plt.show()

h = .02  # step size in the mesh
n_neighbors = 3
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

clf1 = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
clf1.fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)"
          % (n_neighbors))
plt.show()


clf2 = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
clf2.fit(X[etalons], y[etalons])
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[etalons, 0], X[etalons, 1], c=y[etalons], cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)"
          % (n_neighbors))
plt.show()