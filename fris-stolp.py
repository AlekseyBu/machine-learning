import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

iris = datasets.load_wine()

# take the first two features
X = iris.data[:, :2]
y = iris.target
X = np.array([[1,1], [2,2],[1,2], [5,15],[5,9], [5,21], [15,2], [14,3], [16,4]])
y = np.array([0,0,0,1,1,1,2,2,2])

L = 0.4
tetta = 0

def fris(a, b, x):
    # return (numpy.linalg.norm(a - x) - numpy.linalg.norm(a - b)) / numpy.linalg.norm(a - x) + numpy.linalg.norm(a - b))
    return (distance.euclidean(a, x) - distance.euclidean(a, b)) / (distance.euclidean(a, x) + distance.euclidean(a, b))


# return closest to the u object from U
def nearestNeighbor(u, U):
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(U)
    return X[nbrs.kneighbors(u, return_distance=False)]


x = np.array([[1, 1]])
(nearestNeighbor(x, X))


# return new etalon for Y class by existing etalons 'etalons' and set of Xy elements of Y class
def findEtalon(Xy, etalons):
    D, T, E = [], [], []
    i = 0
    xx = []
    for x in Xy:
        xx.append(x)
        #defence
        Xyx = Xy
        Xyx = np.delete(Xyx, i, 0) #Xy without x: Xy\x
        sum = 0
        for u in Xyx:
            sum = sum + fris(u, x, nearestNeighbor(np.reshape(u, (1,-1)), etalons))
        #defence
        D.append(sum/(len(Xy) - 1))

        # tolerance
        sum = 0
        XXy = X[np.all(np.any((X-Xy[:, None]), axis = 2), axis = 0)] #X\Xy
        for v in XXy:
            sum = sum + fris(v, x, nearestNeighbor(np.reshape(v, (1,-1)), etalons))
        #tolerance
        T.append(sum/(len(XXy)))

        # efficiency
        E.append(L*D[i] + (1-L)*T[i])

        i = i + 1
        #print(i)
    #print(xx[np.argmax(E)])
    return xx[np.argmax(E)]

# s = X
# s = np.delete(s,0,0)
# s = np.delete(s,1,0)
# s = np.delete(s,2,0)
# findEtalon(s, X)

def frisStolp():
    #etalon0
    etalon0 = []
    Xy = []
    for i in (np.unique(y)):
        Xy_row = []
        for index, j in enumerate(y):
            if i == j:
                Xy_row.append(X[index,:])
        Xy_row = np.asarray(Xy_row)
        Xy.append(Xy_row)
        XXy = X[np.all(np.any((X - Xy_row[:, None]), axis=2), axis=0)] #X\Xy_row
        etalon0.append(findEtalon(Xy_row, XXy))
    etalon0 = np.asarray(etalon0)

    Xy = np.asarray(Xy)

    #etalons for classes
    etalons = []
    for i in (np.unique(y)):
        etalon_row = []
        union = np.delete(etalon0, i, 0) #delete y row
        etalon_row.append(findEtalon(Xy[i], union))
        etalon_row = np.asarray(etalon_row)
        etalons.append(etalon_row)
    etalons = np.asarray(etalons)

    Xl = X
    #while len(Xl) > 0:
        #correct = []
        #for x in X:
            #val = fris(x, nearestNeighbor(), nearestNeighbor())
            #if val > tetta:
                #correct.append(x)
            #?????????
        #for i in (np.unique(y)):
            #Xy[i] = Xy[i, np.all(np.any((Xy[i] - U), axis=2), axis=0)]  # Xy\U
        #Xl = Xl[np.all(np.any((Xl - U[:, None]), axis=2), axis=0)] #Xl\U
        # for i in (np.unique(y)):
            #union = np.delete(etalons, i, 0)  # delete row with y class
            #etalons[i].append(findEtalon(Xy[i], union))
    return etalons



print(frisStolp())

