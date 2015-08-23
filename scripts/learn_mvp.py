from sklearn import linear_model

X = [[0], [20], [40], [60]]
y = [0,0,0,0]

clf = linear_model.LinearRegression()
clf.fit (X, y)

print(clf.coef_)
print(clf.intercept_)
r2 = clf.score(X,y)
print(r2,'r2')