import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance


# iris = datasets.load_iris()
# # take two features
# X = iris.data[:, [2, 3]]
# y = iris.target

wine = datasets.load_wine()
# take the first two features
X = wine.data[:, [6,9]]
# X = wine.data[:, [5,12]]
y = wine.target

n = len(X)  # data length
xl = np.arange(n)  # data indexes

eps = 1e-5
alpha = 0.7
tetta = 0.1

# substraction of two sets
def sets_diff(a1, a2):
    a1 = np.atleast_1d(a1)
    a2 = np.atleast_1d(a2)
    if (not a1.size):
        return a1
    if (not a2.size):
        return a1
    return np.setdiff1d(a1,a2)

# union of two sets
def sets_union(a1, a2):
    a1 = np.atleast_1d(a1)
    a2 = np.atleast_1d(a2)
    if (not a1.size):
        return a2
    if (not a2.size):
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
    # get etalons values by indexes
    etalons_val = []
    for i in etalons:
        etalons_val.append(X[i])
    etalons_val = np.asarray(etalons_val)
    D, T, E = [], [], []
    i = 0
    arrDiff = sets_diff(xl, Xy)  # X/Xy
    for x in range(length):
        # defence
        sum = 0
        for u in range(length):
            if u == x:
                continue
            sum = sum + fris(X[Xy[u]], X[Xy[x]], nearest_neighbor(np.reshape(X[Xy[u]], (1,-1)), etalons_val))
        # defence
        D.append(sum/(len(Xy) - 1))

        # tolerance
        sum = 0
        for v in range(len(arrDiff)):
            sum = sum + fris(X[arrDiff[v]], X[Xy[x]], nearest_neighbor(np.reshape(X[arrDiff[v]], (1,-1)), etalons_val))
        # tolerance
        T.append(sum/(len(arrDiff)))

        # efficiency
        E.append(alpha*D[i] + (1-alpha)*T[i])

        i = i + 1
    return Xy[np.argmax((E))]  # index from Xy


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
    etalonsUnion = []
    for i in (np.unique(y)):
        etalonsUnion = sets_union(etalonsUnion, etalon0[i])

    etalons = []
    for i in (np.unique(y)):
        etalons.append(find_etalon(xByClass[i], sets_diff(etalonsUnion, etalon0[i])))
    print("etalons:")
    print(etalons)

    # step3: repeat steps4-6 untill X is not empty
    etalonsList = []  # use this trick for comfortable array substaction
    for i in (np.unique(y)):
        etalonsList = sets_union(etalonsList, etalons[i])

    xIndexes = xl
    while (len(xIndexes)):
        # step4: initialize correct obj
        correct = []
        for i in range(len(xIndexes)):
            index = xIndexes[i]  # elemet index
            x = X[index]
            yClass = y[index]

            # values of etalons for specific class
            etalonsYVal = []
            for j in np.atleast_1d(etalons[yClass]):
                etalonsYVal.append(X[j])
            etalonsYVal = np.asarray(etalonsYVal)

            # values of etalons for another classes
            etalons_dif = sets_diff(etalonsList, etalons[yClass])
            etalons_val = []
            for j in np.atleast_1d(etalons_dif):
                etalons_val.append(X[j])
            etalons_val = np.asarray(etalons_val)

            val = fris(x, nearest_neighbor(np.reshape(x, (1,-1)), etalonsYVal), nearest_neighbor(np.reshape(x, (1,-1)), etalons_val))
            if (val > tetta):
                correct.append(index)
        print("correct")
        print(correct)
        if (not len(correct)):
            break

        # step5: delete correct from xByClass and xIndexes
        for i in np.unique(y):
            xByClass[i] = sets_diff(xByClass[i], correct)
        xIndexes = sets_diff(xIndexes, correct)

        # step6: add new etalon for each class
        for i in np.unique(y):
            if (len(xByClass[i])):
                etalons[i] = sets_union(etalons[i], find_etalon(xByClass[i], sets_diff(etalonsList,etalons[i])))

        etalonsList = []
        for i in np.unique(y):
            etalonsList = sets_union(etalonsList, etalons[i])

        print(etalons)

    return etalons

# ans = fris_stolp()
# ans = [39, [66, 77, 98], [101, 106, 119]]
# ans =  [[ 8, 21, 43], [ 73,  74,  78,  81,  86,  87,  89, 112, 129], [133, 142, 150, 151, 175]]
ans = [[35, 38, 49], [ 61,  65,  66,  98, 113, 121, 126], [145, 162, 175]]
print("final etalons:")
print(ans)

etalons = []
for i in np.unique(y):
    etalons = sets_union(etalons, ans[i])
etalons_persentage = 100*len(etalons)/len(y)

def print_plot(X, Y, dataset, xlabel, ylabel, features, check, x_test, y_test) :
    colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.scatter(X[:,0], X[:,1], c=y, cmap=colors, s=30)
    plt.scatter(X[etalons,0], X[etalons,1], c=y[etalons], cmap=colors, s=300)
    plt.title("Etalons (%.2f%% of all dataset) for %s data set with alpha = %f" %
              (etalons_persentage, dataset, alpha))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    red_patch = mpatches.Patch(color="red", label=features[0])
    green_patch = mpatches.Patch(color="green", label=features[1])
    blue_patch = mpatches.Patch(color="blue", label=features[2])
    plt.legend(handles=[red_patch, green_patch, blue_patch])

    if (check):
        colors = ListedColormap(['#FFFF00', '#FFFF00', '#FFFF00'])
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=colors, s=30)
    plt.show()

# print_plot(X, y, 'iris flower', 'Sepal Length', 'Sepal Width', ['Setosa', 'Versicolor', 'Virginica'], False, None, None)
# print_plot(X, y, 'wine', 'Phenols', 'Proline', ['Class1', 'Class2', 'Class3'], False, None, None)
print_plot(X, y, 'wine', 'Flavonoids', 'Color int', ['Class1', 'Class2', 'Class3'], False, None, None)

# calculate quality

# test data for iris
# x1 = [[1, 0.1], [1.23, 1.3], [1.5, 0.4], [2.62, 0.4], [1.37, 0.8]]
# x2 = [[3.17, 0.89], [4.1, 1.1], [3.5, 1.7], [4.9, 1.26], [5.5, 1.2]]
# x3 = [[6, 2], [7, 1.6], [6.5, 1.5], [5, 2], [4.5, 2.2]]
# y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]


# # test data for wine
# x1 = [[3.5,1400], [2.5,1202], [3,1000], [3.8,1210], [3.3,1300]]
# x2 = [[1.8,300], [2.0, 400], [2.8, 400], [3.5, 600], [3.8, 700]]
# x3 = [[1.2, 380], [1.3, 700], [1.4, 780], [1.5, 650], [1.7, 830]]
# y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

# test data for wine
x1 = [[3.5, 9], [2.5, 8], [3, 5], [4, 8.3], [3.3, 10]]
x2 = [[1, 2], [2, 3], [2.8, 2], [3.5, 3], [5, 4]]
x3 = [[1, 10], [1.3, 12], [0.5, 8], [1.8, 9], [0.3, 4]]
y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

x_test = x1+x2+x3
x_test = np.array(x_test)

# print_plot(X, y, 'iris flower', 'Sepal Length', 'Sepal Width', ['Setosa', 'Versicolor', 'Virginica'], True, x_test, y_test)
# print_plot(X, y, 'wine', 'Phenols', 'Proline', ['Class1', 'Class2', 'Class3'], True, x_test, y_test)
print_plot(X, y, 'wine', 'Flavonoids', 'Color int', ['Class1', 'Class2', 'Class3'], True, x_test, y_test)

def calc_quality(X, y, x_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X, y)
    sum = len(x_test)
    for i, val in enumerate(x_test):
        # print(neigh.predict(np.reshape(val, (1,2))), y_test[i])
        # print(neigh.predict_proba(np.reshape(val, (1, 2))))
        if (neigh.predict(np.reshape(val, (1,2))) != y_test[i]):
            sum = sum - 1
    quality = sum/len(x_test)
    return quality

# for all dataset
quality = calc_quality(X, y, x_test, y_test)
print("Classification quality for all dataset = %.2f" % quality)

# for etalons
x_etalons = []
y_etalons = []
for val in etalons:
    x_etalons.append([X[val, 0], X[val, 1]])
    y_etalons.append(y[val])

quality = calc_quality(x_etalons, y_etalons, x_test, y_test)
print("Classification quality for etalons dataset = %.2f" % quality)

