# Author: 无名的弗兰克 @ ChemDog
# https://github.com/Frank-the-Obscure/
# regresssor for weibo data
# -*- coding: utf-8 -*-


def init(mode):
    """ 整理为可直接用于回归的 X, y, weight

    i: features
    o: X, y, weight(return)
    """
    import scipy.io as sio
    import scipy as sp


    combined_list = [
        sp.sparse.csc_matrix(sio.loadmat('08-12-cut_uid_ave.mat')['X']),
        sp.sparse.csc_matrix(sio.loadmat('08-12-cut_X_length.mat')['X']),

    ]
    if mode == 'f':
        combined_list.append(sio.loadmat('08-12-cut_Xf.mat')['X'])
    elif mode == 'c':
        combined_list.append(sio.loadmat('08-12-cut_Xc.mat')['X'])
    else:
        combined_list.append(sio.loadmat('08-12-cut_Xl.mat')['X'])

    X = sp.sparse.hstack(combined_list)

    if mode == 'f':
        y = sio.loadmat('08-12-cut_y_forward.mat')['y'] #
    elif mode == 'c':
        y = sio.loadmat('08-12-cut_y_comment.mat')['y'] 
    else:
        y = sio.loadmat('08-12-cut_y_like.mat')['y']

    weight = sio.loadmat('08-12-cut_weight.mat')['weight']

    print(mode, X.shape, y.ravel().shape)
    return X, y.ravel(), weight.ravel()

def init2():
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
        X, y, weight, test_size=0.4, random_state=0)
    clf = RandomForestRegressor(n_estimators=20, max_features='sqrt', n_jobs=-1)
    clf.fit(X_train, y_train, weight_train)
    print(clf.score(X_test, y_test, weight_test))

    #clf = RandomForestRegressor(n_estimators=10, max_features='sqrt', n_jobs=-1)
    #clf.fit(X, y, weight)

    #from sklearn.externals import joblib
    #joblib.dump(clf, 'models/random_forest_8-12_like.pkl') 

def gbrt_regressor(X, y, weight):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn import cross_validation
    
    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.4, random_state=0)
    clf = GradientBoostingRegressor(n_estimators=100, max_features='sqrt')
    clf.fit(X_train, y_train, weight_train)
    print(clf.score(X_test, y_test, weight_test))

    #clf = RandomForestRegressor(n_estimators=10, max_features='sqrt', n_jobs=-1)
    #clf.fit(X, y, weight)

    #from sklearn.externals import joblib
    #joblib.dump(clf, 'models/random_forest_8-12_like.pkl') 

def random_forest_regressor2(X, y, weight):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import cross_validation

    '''
    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.1, random_state=0)
    clf = RandomForestRegressor(n_estimators=10, max_features='sqrt', n_jobs=-1)
    clf.fit(X_train, y_train, weight_train)
    print(clf.score(X_test, y_test, weight_test))'''

    clf = RandomForestRegressor(n_estimators=10, max_features='sqrt', n_jobs=-1)
    clf.fit(X, y, weight)

    from sklearn.externals import joblib
    joblib.dump(clf, 'models/random_forest_like_all.pkl') 

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
        sp.sparse.csc_matrix(sio.loadmat('07-cut_uid_ave.mat')['X']),
        sp.sparse.csc_matrix(sio.loadmat('07-cut_X_length.mat')['X']),
        sio.loadmat('07-cut_Xf.mat')['X']
    ]

    X = sp.sparse.hstack(combined_list)

    print(X.shape)
    return X

def random_forest_predictor(X):
    """ 预测器

    """
    from sklearn.externals import joblib
    import scipy.io as sio

    clf = joblib.load('models/random_forest_8-12_forward.pkl') 
    y = clf.predict(X)
    print(y.shape)

    sio.savemat('12-forward.mat', {'y':y})

def log_reg_test(X, y, weight):
    from sklearn.linear_model import LogisticRegression
    from sklearn import cross_validation
    from sklearn.metrics import confusion_matrix

    weights = {0:1, 1:3, 2:7, 3:14, 4:30, 5:70, 6:101}

    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.2, random_state=0)
    clf = LogisticRegression(class_weight=weights, C=0.01, solver='liblinear',max_iter=20)
    #clf = LogisticRegression( max_iter=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(confusion_matrix(y_test, y_pred))

def naive_bayes(X, y, weight):
    from sklearn.naive_bayes import GaussianNB
    from sklearn import cross_validation
    from sklearn.metrics import confusion_matrix

    weights = {0:1, 1:3, 2:7, 3:14, 4:30, 5:70, 6:101}

    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.2, random_state=0)
    clf = GaussianNB()
    #clf = LogisticRegression( max_iter=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(confusion_matrix(y_test, y_pred))

def rfc(X, y, weight):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import cross_validation
    from sklearn.metrics import confusion_matrix

    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(n_estimators=20, max_features='sqrt', n_jobs=-1)
    clf.fit(X_train, y_train, weight_train)
    y_pred = clf.predict(X_test)

    print(confusion_matrix(y_test, y_pred))

def sgd(X_train, y_train, weight, X_test):
    from sklearn.linear_model import SGDRegressor
    from sklearn import cross_validation
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler


    clf = SGDRegressor(loss="epsilon_insensitive", n_iter=500, penalty="l2")
    #clf = LogisticRegression( max_iter=100)

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)  # apply same transformation to test data

    clf.fit(X_train, y_train, sample_weight=weight)

    y_pred = clf.predict(X_test)

    from sklearn.externals import joblib
    import scipy.io as sio
    joblib.dump(clf, 'models/sgd_like.pkl') 
    sio.savemat('07_y_forward.mat', {'y':y_pred})

def sgd_test(X, y, weight):
    from sklearn.linear_model import SGDClassifier
    from sklearn import cross_validation
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test, weight_train, weight_test = cross_validation.train_test_split(
        X, y, weight, test_size=0.2, random_state=0)
    clf = SGDClassifier(loss="hinge", n_iter=20, n_jobs=-1, penalty="l2")
    #clf = LogisticRegression( max_iter=100)

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)  # apply same transformation to test data

    clf.fit(X_train, y_train, sample_weight=weight_train)
    y_pred = clf.predict(X_test)

    print(confusion_matrix(y_test, y_pred))

def predict(filein, filename):
    # load predict data
    import scipy.io as sio
    import json

    predict_forward = sio.loadmat(filename + '_forward.mat')['y']
    predict_comment = sio.loadmat(filename + '_comment.mat')['y']
    predict_like = sio.loadmat(filename + '_like.mat')['y']

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
        (uid, mid, day, forward_count,
        comment_count, like_count, content) = json.loads(line)
        #(uid, mid, day, content) = json.loads(line)
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
    X, y, weight = init('f')
    X = X.tocsr()
    print(type(X))
    X_test = init_predict()
    X_test = X_test.tocsr()
    #print(X.shape)
    sgd(X, y, weight, X_test)'''
    #rfc(X[:500000], y[:500000], weight[:500000])
    #X, y, weight = init2()
    #random_forest_regressor2(X, y, weight)

    #random_forest_predictor(X)
    
    filein = open('07-cut.txt', encoding='utf-8')
    predict(filein, '07_y')
    
    t1 = time.time()
    print('Finished: runtime {}'.format(t1 - t0))
    #cut_replace(filein)

if __name__ == '__main__':
    main()