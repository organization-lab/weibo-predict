# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# regresssor for weibo data
# -*- coding: utf-8 -*-


def init():
    """ 整理为可直接用于回归的 X, y, weight

    i: features
    o: X, y, weight(return)
    """
    import scipy.io as sio
    import scipy as sp


    combined_list = [
        sp.sparse.csc_matrix(sio.loadmat('weibo_train_data_cut_uid_ave.mat')['X']),
        sp.sparse.csc_matrix(sio.loadmat('weibo_train_data_cut_X_length.mat')['X']),
        sio.loadmat('weibo_train_data_cut_X_like_dict.mat')['X']
    ]

    X = sp.sparse.hstack(combined_list)

    y = sio.loadmat('weibo_train_data_cut_y_like.mat')['y']
    weight = sio.loadmat('weibo_train_data_cut_weight.mat')['weight']

    print(X.shape, y.ravel().shape)
    return X, y.ravel(), weight.ravel()

def linear_regressor(X, y, weight):
    from sklearn import linear_model
    from sklearn import cross_validation

    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.4, random_state=0)
    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train, n_jobs=-1)
    print(clf.score(X_test, y_test, weight_test))

def linear_regressor_bench(X, y, weight):
    from sklearn import linear_model
    from sklearn import cross_validation

    #X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
    #    X, y, weight, test_size=0.1, random_state=0)
    clf = linear_model.LinearRegression()
    clf.fit(X, y, n_jobs=-1)
    print(clf.score(X, y, weight))

    from sklearn.externals import joblib
    joblib.dump(clf, 'linear_regressor_4.pkl') 

def random_forest_regressor(X, y, weight):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import cross_validation

    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.1, random_state=0)
    clf = RandomForestRegressor(n_estimators=10, max_features='sqrt', n_jobs=-1)
    clf.fit(X_train, y_train, weight_train)
    print(clf.score(X_test, y_test, weight_test))

    from sklearn.externals import joblib
    joblib.dump(clf, 'models/random_forest_like2.pkl') 

def svr(X, y, weight):
    """SVR 

    """

    from sklearn import svm
    from sklearn import cross_validation

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.4, random_state=0)

    # Set regularization parameter
    # turn down tolerance for short training time
    clf = svm.SVR(kernel='linear')

    clf.fit(X, y)
    print(clf.score(X, y))

def init_predict():
    """ 整理为用于预测的 X

    i: features
    o: X
    """
    import scipy.io as sio
    import scipy as sp


    combined_list = [
        sp.sparse.csc_matrix(sio.loadmat('weibo_predict_data_cut_uid_ave.mat')['X']),
        sp.sparse.csc_matrix(sio.loadmat('weibo_predict_data_cut_X_length.mat')['X']),
        sio.loadmat('weibo_predict_data_cut_Xl.mat')['X']
    ]

    X = sp.sparse.hstack(combined_list)

    print(X.shape)
    return X

def random_forest_predictor(X):
    """ 预测器

    """
    from sklearn.externals import joblib
    import scipy.io as sio

    clf = joblib.load('models/random_forest_like2.pkl') 
    y = clf.predict(X)
    print(y.shape)

    sio.savemat('weibo_predict_data-like.mat', {'y':y})

def predict(filein, filename):
    # load predict data
    import scipy.io as sio
    import json

    predict_forward = sio.loadmat(filename + '-forward.mat')['y']
    predict_comment = sio.loadmat(filename + '-comment.mat')['y']
    predict_like = sio.loadmat(filename + '-like.mat')['y']

    predict_forward = predict_forward.ravel()
    predict_comment = predict_comment.ravel()
    predict_like = predict_like.ravel()

    def process(num):
        if num < 0:
            num = 0
        else:
            num = round(float(num))
            #print(type(num))
        return num

    fileout = open(filename + 'predict.txt', 'w', encoding='utf-8')
    # load file
    i = 0
    for line in filein:
        #(uid, mid, day, forward_count,
        #comment_count, like_count, content) = json.loads(line)
        (uid, mid, day, content) = json.loads(line)

        #print(predict_forward[i], predict_comment[i], predict_like[i])

        predict_forward_cut = process(predict_forward[i])
        predict_comment_cut = process(predict_comment[i])
        predict_like_cut = process(predict_like[i])

        #print(predict_forward_cut, predict_comment_cut, predict_like_cut)
        predict_line = str(predict_forward_cut) + ',' + str(predict_comment_cut) + ',' +  str(predict_like_cut)
        fileout.write('\t'.join([uid, mid, predict_line]) + '\n')
        i += 1
        #if i == 10:
        #    break

def main():
    import time
    t0 = time.time()
    
    '''
    X, y, weight= init()
    
    X = X[:1000000]
    y = y[:1000000]
    weight = weight[:1000000]

    print(X.shape)

    #linear_regressor(X, y, weight)
    random_forest_regressor(X, y, weight)
    '''

    #X = init_predict()
    #random_forest_predictor(X)
    
    filein = open('weibo_predict_data_cut.txt', encoding='utf-8')
    predict(filein, 'weibo_predict_data')
    
    t1 = time.time()
    print('Finished: runtime {}'.format(t1 - t0))
    #cut_replace(filein)

if __name__ == '__main__':
    main()