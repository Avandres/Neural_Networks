from sklearn import decomposition
import numpy

class PCA():

    def create_mas(self, mas, j):
        newmas = []
        for i in range(len(mas[0])):
            newmas.append(mas[j][i])
        return newmas

    def max_element(self, mas):
        max = [0, mas[0]]
        for i in range(len(mas)):
            if mas[i] > max[1]:
                max = [i, mas[i]]
        return max

    def min_element(self, mas):
        min = [0, mas[0]]
        for i in range(len(mas)):
            if mas[i] < min[1]:
                min = [i, mas[i]]
        return min

    def Center(self, X_train):
        for i in range(len(X_train)):
            mas = self.create_mas(X_train, i)
            minEl = self.min_element(mas)
            maxEl = self.max_element(mas)
            Tab = minEl[1] + ((maxEl[1] - minEl[1]) / 2)
            for j in range(len(X_train[0])):
                X_train[i][j] = X_train[i][j] - Tab
        return X_train


    def cov(self, X, Y):
        cov = 0
        Xmid = 0
        Ymid = 0
        for i in range(len(X)):
            Xmid += X[i]
            Ymid += Y[i]
        Xmid = Xmid/len(X)
        Ymid = Ymid/len(Y)
        for i in range(len(X)):
            cov += (X[i] - Xmid)*(Y[i] - Ymid)
        cov = cov/len(X)
        return cov


    def Cov_matrix(self, X_train):
        cov_matrix = []
        for i in range(len(X_train[0])):
            Xi = []
            for k in range(len(X_train)):
                Xi.append(X_train[k][i])
            cov_matrix.append([])
            for j in range(len(X_train[0])):
                Xj = []
                for k in range(len(X_train)):
                    Xj.append(X_train[k][j])
                cov_matrix[i].append(self.cov(Xi, Xj))
        return cov_matrix

    def Assessment(self, X_train):
        matrix = self.Cov_matrix(X_train)
        eigenvalues, eigenvectors = numpy.linalg.eig(matrix)
        s = 0
        for i in range(len(eigenvalues)):
            s += eigenvalues[i]
        s2 = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == j:
                    s2 += matrix[i][j]
        return s2/s

    def Proect(self, X_train):
        #matrix = self.Cov_matrix(X_train)
        matrix = numpy.cov(X_train)
        eigenvalues, eigenvectors = numpy.linalg.eig(matrix)
        i = self.max_element(eigenvalues)
        i = i[0]
        MaxDisperseVector = eigenvectors[i]
        V = MaxDisperseVector
        V = -V
        New_X_train = numpy.dot(V, X_train)
        return New_X_train

x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[15, 20, 25, 30, 35, 40, 45, 50, 55, 60]]
X_train = numpy.asarray(x)

skPCA = decomposition.PCA(n_components=1)
myPCA = PCA()
skX = skPCA.fit_transform(numpy.transpose(X_train))
X = myPCA.Center(X_train)
X = myPCA.Proect(X)
print(skX)
print()
print(X)
