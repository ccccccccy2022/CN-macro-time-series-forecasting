from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
import pandas as pd
import math


# 几个准确率的评估函数
def makescore(n_for):
    def direction_accuracy(y_true, y_predict):
        def direction(c):
            yt = pd.DataFrame(np.zeros((1, len(y_true) - c)))
            yp = pd.DataFrame(np.zeros((1, len(y_predict) - c)))
            for i in range(0, len(y_true) - c, 1):
                if y_true[i + c] - y_true[i] >= 0:
                    yt[i] = 1
                else:
                    yt[i] = 0
                if y_predict[i + c] - y_predict[i] >= 0:
                    yp[i] = 1
                else:
                    yp[i] = 0
            return yt, yp

        Yt = np.zeros((1, 0))
        Yp = np.zeros((1, 0))
        for j in range(1, len(y_true), 1):
            Yt = np.append(Yt, direction(j)[0])
            Yp = np.append(Yp, direction(j)[1])
        score = accuracy_score(Yt, Yp)
        return score

    def dv_accuracy(y_true, y_predict):
        def direction(c):
            yt = pd.DataFrame(np.zeros((1, len(y_true) - c)))
            yp = pd.DataFrame(np.zeros((1, len(y_predict) - c)))
            for i in range(0, len(y_true) - c, 1):
                if y_true[i + c] - y_true[i] >= 0:
                    yt[i] = 1
                else:
                    yt[i] = 0
                if y_predict[i + c] - y_predict[i] >= 0:
                    yp[i] = 1
                else:
                    yp[i] = 0
            return yt, yp

        def MSE(y, t):
            return 1 / len(y) * np.sum((y - t) ** 2)

        Yt = np.zeros((1, 0))
        Yp = np.zeros((1, 0))
        for j in range(1, len(y_true), 1):
            Yt = np.append(Yt, direction(j)[0])
            Yp = np.append(Yp, direction(j)[1])
        score = math.exp(accuracy_score(Yt, Yp)) - math.exp(MSE(y_true, y_predict))
        return score

    def dv2_accuracy(y_true, y_predict):
        # global n_for

        def direction(c):
            yt = pd.DataFrame(np.zeros((1, len(y_true) - c)))
            yp = pd.DataFrame(np.zeros((1, len(y_predict) - c)))
            for i in range(0, len(y_true) - c, 1):
                if y_true[i + c] - y_true[i] > 0:
                    yt[i] = 1
                elif y_true[i + c] - y_true[i] < 0:
                    yt[i] = 0
                else:
                    yt[i] = -1
                if y_predict[i + c] - y_predict[i] > 0:
                    yp[i] = 1
                elif y_predict[i + c] - y_predict[i] < 0:
                    yp[i] = 0
                else:
                    yp[i] = -1

            return yt, yp

        def MSE(y, t):
            return 1 / len(y) * np.sum((y - t) ** 2)

        Yt = np.zeros((1, 0))
        Yp = np.zeros((1, 0))
        for j in range(1, len(y_true), 1):
            Yt = np.append(Yt, direction(j)[0])
            Yp = np.append(Yp, direction(j)[1])

        def werror(y_true, y_predict):
            weightstart = 90
            weightend = 10
            weight = np.linspace(weightstart, weightend, n_for)
            we = np.zeros((1, 1))
            for i in range(0, len(y_true), 1):
                we = weight[i] * abs(y_true[i] - y_predict[i]) + we
            we = we / np.sum(weight)
            return we

        score = math.exp(accuracy_score(Yt, Yp)) - math.exp(20 * werror(y_true, y_predict))
        return score

    def dv3_accuracy(y_true, y_predict):
        import sympy
        # global n_for
        n_extend = 100

        def direction(c):
            yt = pd.DataFrame(np.zeros((1, len(y_true) - c)))
            yp = pd.DataFrame(np.zeros((1, len(y_predict) - c)))
            for i in range(0, len(y_true) - c, 1):
                if y_true[i + c] - y_true[i] > 0:
                    yt[i] = 1
                elif y_true[i + c] - y_true[i] < 0:
                    yt[i] = 0
                else:
                    yt[i] = -1
                if y_predict[i + c] - y_predict[i] > 0:
                    yp[i] = 1
                elif y_predict[i + c] - y_predict[i] < 0:
                    yp[i] = 0
                else:
                    yp[i] = -1

            return yt, yp

        def MSE(y, t):
            return 1 / len(y) * np.sum((y - t) ** 2)

        Yt = np.zeros((1, 0))
        Yp = np.zeros((1, 0))
        for j in range(1, len(y_true), 1):
            Yt = np.append(Yt, direction(j)[0])
            Yp = np.append(Yp, direction(j)[1])

        def werror(y_true, y_predict):
            weightstart = 90
            weightend = 10
            weight = np.linspace(weightstart, weightend, n_for)
            we = np.zeros((1, 1))
            for i in range(0, len(y_true), 1):
                we = weight[i] * abs(y_true[i] - y_predict[i]) + we
            we = we / np.sum(weight)
            return we

        def regress(M, N, x_n, t_n, lamda=0):
            print("-----------------------M=%d, N=%d-------------------------" % (M, N))
            order = np.arange(M + 1)
            order = order[:, np.newaxis]
            e = np.tile(order, [1, N])
            XT = np.power(x_n, e)
            X = np.transpose(XT)
            a = np.matmul(XT, X) + lamda * np.identity(M + 1)  # X.T * X
            b = np.matmul(XT, t_n)  # X.T * T
            w = np.linalg.solve(a, b)  # aW = b => (X.T * X) * W = X.T * T
            # print("W:")
            # print(w)
            # e2 = np.tile(order, [1,x_n.shape[0]])
            # XT2 = np.power(x, e2)
            p = np.matmul(w.reshape(1, len(w)), XT)
            orderdiff = np.arange(M)
            orderdiff = orderdiff[:, np.newaxis]
            x = sympy.symbols("x")
            diffx = np.power(x, orderdiff).reshape(M, 1)
            y = np.matmul(w[1:].reshape(1, len(w[1:])), np.arange(1, 6, 1).reshape(M, 1) * diffx) + w[0]
            return p, w, XT, y

        def extend(y1, n_extend):
            # global n_for
            shortline = []
            longline = []
            shortline = list(map(lambda i: y1[i:i + 2], range(n_for - 1)))
            longline = list(map(lambda i: np.linspace(shortline[i][0], shortline[i][1], n_extend) \
                                , range(np.array(shortline).shape[0])))
            longline2 = np.array(longline).flatten()
            return shortline, longline2

        def pointextrema(y1):
            t_n = extend(y1, 100)[1].flatten()
            N = len(t_n)
            x_n = np.linspace(0, 1, N)
            M = 5
            p = regress(M, N, x_n, t_n, lamda=0)[0].flatten()
            w = regress(M, N, x_n, t_n, lamda=0)[1]
            y = regress(M, N, x_n, t_n, lamda=0)[3]
            return p, w, y

        def point(y1, nalpha):
            y = pointextrema(y1)[2][0][0]
            x = sympy.symbols("x")
            result = pd.DataFrame(
                list(map(lambda i: y.subs({x: np.linspace(0, 1, 1100)[i]}), range((n_for - 1) * n_extend))))
            result = result[result > -nalpha]
            result = result[result < nalpha]
            index = result.dropna().index
            return index

        index1 = point(y_true, 0.1)
        index2 = point(y_predict, 0.1)
        a = set(index1).intersection(set(index2))
        if len(a) > 2 / 3 * len(index1):
            score = math.exp(accuracy_score(Yt, Yp) + 1) - math.exp(20 * werror(y_true, y_predict))
        elif len(a) > 1 / 3 * len(index1):
            score = math.exp(accuracy_score(Yt, Yp) + 0.5) - math.exp(20 * werror(y_true, y_predict))
        else:
            score = math.exp(accuracy_score(Yt, Yp)) - math.exp(20 * werror(y_true, y_predict))
        return score

    def dv4_accuracy(y_true, y_predict):
        if len(y_true) == 1:
            score = -abs(y_true[0] - y_predict[0])
        else:
            k_true = np.zeros((len(y_true) - 1, 1))
            k_predict = np.zeros((len(y_true) - 1, 1))
            K_e = 0
            MAE = abs(y_true[0] - y_predict[0])
            for i in range(0, len(y_true) - 1, 1):
                k_true[i] = y_true[i + 1] - y_true[i]
                k_predict[i] = y_predict[i + 1] - y_predict[i]
                if k_true[i] * k_predict[i] > 0:
                    MAE = MAE + abs(y_true[i] - y_predict[i])
                    # deltaMAE=deltaMAE+abs(abs(y_true[i+1]-y_predict[i+1])-abs(y_true[i]-y_predict[i]))
                    K_e = K_e + abs(k_true[i] - k_predict[i])
                else:
                    MAE = MAE + 1
                    K_e = K_e + 2
            # score=-(K_e^2+MAE^2)
            score = -(np.exp(K_e) ** (1 / 2) + np.exp(MAE) ** (1 / 4))
        return score

    def dv5_accuracy(y_true, y_predict):
        if len(y_true) == 1:
            score = -abs(y_true[0] - y_predict[0])
        else:
            k_true = np.zeros((len(y_true) - 1, 1))
            k_predict = np.zeros((len(y_true) - 1, 1))
            K_e = 0
            MAE = abs(y_true[0] - y_predict[0])
            for i in range(0, len(y_true) - 1, 1):
                k_true[i] = y_true[i + 1] - y_true[i]
                k_predict[i] = y_predict[i + 1] - y_predict[i]
                if k_true[i] * k_predict[i] > 0:
                    MAE = MAE + abs(y_true[i] - y_predict[i])
                    # deltaMAE=deltaMAE+abs(abs(y_true[i+1]-y_predict[i+1])-abs(y_true[i]-y_predict[i]))
                    K_e = K_e + abs(k_true[i] - k_predict[i])
                else:
                    MAE = MAE + 10
                    K_e = K_e + 20
            score = -(K_e ** 2 + MAE ** 2)
        return score

    def dv6_accuracy(y_true, y_predict):
        if len(y_true) == 1:
            score = -abs(y_true[0] - y_predict[0])
        else:
            k_true = np.zeros((len(y_true) - 1, 1))
            k_predict = np.zeros((len(y_true) - 1, 1))
            K_e = 0
            MAE = abs(y_true[0] - y_predict[0])
            for i in range(0, len(y_true) - 1, 1):
                k_true[i] = y_true[i + 1] - y_true[i]
                k_predict[i] = y_predict[i + 1] - y_predict[i]
                if k_true[i] * k_predict[i] > 0:
                    MAE = MAE + abs(y_true[i] - y_predict[i])
                    # deltaMAE=deltaMAE+abs(abs(y_true[i+1]-y_predict[i+1])-abs(y_true[i]-y_predict[i]))
                    K_e = K_e + abs(k_true[i] - k_predict[i])
                else:
                    MAE = MAE + 10
                    K_e = K_e + 200
            score = -(K_e + MAE)
        return score

    def dv7_accuracy(y_true, y_predict):
        if len(y_true) == 1:
            score = -abs(y_true[0] - y_predict[0])
        else:
            k_true = np.zeros((len(y_true) - 1, 1))
            k_predict = np.zeros((len(y_true) - 1, 1))
            K_e = 0
            MAE = abs(y_true[0] - y_predict[0])
            for i in range(0, len(y_true) - 1, 1):
                k_true[i] = y_true[i + 1] - y_true[i]
                k_predict[i] = y_predict[i + 1] - y_predict[i]
                if k_true[i] * k_predict[i] > 0:
                    MAE = MAE + abs(y_true[i] - y_predict[i])
                    # deltaMAE=deltaMAE+abs(abs(y_true[i+1]-y_predict[i+1])-abs(y_true[i]-y_predict[i]))
                    K_e = K_e + abs(k_true[i] - k_predict[i])
                else:
                    MAE = MAE + 1
                    K_e = K_e + 2
            # score=-(K_e^2+MAE^2)
            score = -(np.exp(K_e) ** (1 / 4) + np.exp(MAE) ** (1 / 2))
        return score

    def dv8_accuracy(y_true, y_predict):
        A = 8
        if len(y_true) == 1:
            score = -abs(y_true[0] - y_predict[0])
        else:
            k_true = np.zeros((len(y_true) - 1, 1))
            k_predict = np.zeros((len(y_true) - 1, 1))
            K_e = 0
            MAE = abs(y_true[0] - y_predict[0])
            for i in range(0, len(y_true) - 1, 1):
                k_true[i] = y_true[i + 1] - y_true[i]
                k_predict[i] = y_predict[i + 1] - y_predict[i]
                if k_true[i] * k_predict[i] > 0:
                    MAE = MAE + max(abs(y_true[i] - y_predict[i]), abs(y_true[i + 1] - y_predict[i + 1]))
                    # deltaMAE=deltaMAE+abs(abs(y_true[i+1]-y_predict[i+1])-abs(y_true[i]-y_predict[i]))
                    K_e = K_e + abs(k_true[i] - k_predict[i])
                else:
                    MAE = MAE + max(abs(y_true[i] - y_predict[i]), abs(y_true[i + 1] - y_predict[i + 1]))
                    K_e = K_e + A * abs(k_true[i] - k_predict[i])
            score = -(K_e + MAE)
            # score=-(np.exp(K_e)**(1/4)+np.exp(MAE)**(1/2))
        return score

    direction_score = make_scorer(direction_accuracy, greater_is_better=True)  # 同roc_auc_score方法相同，输出结果为类别的概率值
    dv_score = make_scorer(dv_accuracy, greater_is_better=True)
    dv2_score = make_scorer(dv2_accuracy, greater_is_better=True)
    dv3_score = make_scorer(dv3_accuracy, greater_is_better=True)
    dv4_score = make_scorer(dv4_accuracy, greater_is_better=True)
    dv5_score = make_scorer(dv5_accuracy, greater_is_better=True)
    dv6_score = make_scorer(dv6_accuracy, greater_is_better=True)
    dv7_score = make_scorer(dv7_accuracy, greater_is_better=True)
    dv8_score = make_scorer(dv8_accuracy, greater_is_better=True)
    scoring = {'Dv': dv_score, 'Dv2': dv2_score, 'Dv3': dv3_score, 'Dv4': dv4_score, 'Dv5': dv5_score, 'Dv6': dv6_score,
               'Dv7': dv7_score, 'Dv8': dv8_score}
    return scoring


def direction_accuracy(y_true, y_predict):
    def direction(c):
        yt = pd.DataFrame(np.zeros((1, len(y_true) - c)))
        yp = pd.DataFrame(np.zeros((1, len(y_predict) - c)))
        for i in range(0, len(y_true) - c, 1):
            if y_true[i + c] - y_true[i] >= 0:
                yt[i] = 1
            else:
                yt[i] = 0
            if y_predict[i + c] - y_predict[i] >= 0:
                yp[i] = 1
            else:
                yp[i] = 0
        return yt, yp

    Yt = np.zeros((1, 0))
    Yp = np.zeros((1, 0))
    for j in range(1, len(y_true), 1):
        Yt = np.append(Yt, direction(j)[0])
        Yp = np.append(Yp, direction(j)[1])
    score = accuracy_score(Yt, Yp)
    return score


def train_on_trainset(scoring, point_line_flag, train_x2, train_y2, test_x2, test_y, nnn, DVstandard):
    best_estimator = list(np.zeros((nnn, 1)))
    ypred = list(np.zeros((nnn, 1)))
    ypredt = list(np.zeros((nnn, 1)))
    score = list(np.zeros((nnn, 1)))
    fit = list(np.zeros((nnn, 1)))
    for j in range(0, nnn, 1):
        if point_line_flag == 1:
            tss = TimeSeriesSplit(n_splits=np.int(train_x2[j].shape[0] / test_y[j].shape[0]) - 1)
        else:
            tss = TimeSeriesSplit(n_splits=2)
    for j in range(0, nnn, 1):
        X = np.array(train_x2[j])
        y = train_y2[j]
        grid = GridSearchCV(estimator=bst, param_grid=param_dist,
                            cv=tss.split(X, y), scoring=scoring[DVstandard], refit='Dv4',
                            verbose=1, n_jobs=6)
        fit[j] = grid.fit(X, y)
        best_estimator[j] = grid.best_estimator_
        ypred[j] = best_estimator[j].predict(np.array(test_x2[j]))
        ypredt[j] = best_estimator[j].predict(np.array(train_x2[j]))
        score[j] = direction_accuracy(np.array(test_y[j]), ypred[j])
    return fit, best_estimator, score, ypredt, ypred


def Fourierypred(ypred, error):
    ypredd = list(np.zeros((nnn, 1)))
    for j in range(0, nnn, 1):
        ypredd[j] = ypred[j] + error[j].flatten()
    return ypredd
