from sklearn.preprocessing import Binarizer
X = [[ 50, 50,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
transformer = Binarizer().fit(X)  # fit does nothing.
transformer
transformer.transform(X)

print(transformer.transform(X))


