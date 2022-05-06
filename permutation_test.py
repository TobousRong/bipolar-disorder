from sklearn import linear_model
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_predict
import multiprocessing as mp
from scipy.io import savemat
from tqdm import tqdm
from pylab import *
import numpy as np

def LOO_CV(feature, score):
    a = int(len(score))
    reg = linear_model.LinearRegression()
    F, pval = f_regression(feature, score)
    coe1 = [];
    for num in range(1, len(F), 1):
        xnew = SelectKBest(f_regression, k=num).fit_transform(feature, score)
        scores = cross_val_predict(reg, xnew, score, cv=a)
        cor = np.corrcoef(score, scores)
        coe1.append(cor[0,1])
    coe2 = [];
    xnew = SelectKBest(f_regression, k=coe1.index(max(coe1)) + 1).fit_transform(feature,score)
    scores = cross_val_predict(reg, xnew,score, cv=a)
    cor = np.corrcoef(scores, score)
    coe2.append(cor[0, 1])
    return xnew,reg,coe2

def permutation_test(xnew,reg,score,f):
    coe3=[];
    a = int(len(score))
    np.random.seed(f)
    score_new = np.random.permutation(score)
    scores = cross_val_predict(reg, xnew, score_new, cv=a)
    cor = np.corrcoef(score_new, scores)
    coe3.append(cor[0,1])
    return coe3

if __name__ == '__main__':
    inputfile = 'D:/Desktop/study/bip/manuscript_bip/data/pc.txt'
    feature = np.loadtxt(inputfile)
    inputfile = 'D:/Desktop/study/bip/manuscript_bip/data/adhd.txt'
    score = np.loadtxt(inputfile)
    # inputfile = 'D:/Desktop/data/bipolar.txt'
    # other = np.loadtxt(inputfile)


    xnew,reg,coe2=LOO_CV(feature,score)
    X=[]
    result = []
    permutation_num = 9999
    pbar = tqdm(total=permutation_num)
    pbar.set_description('LOO_CV')
    update = lambda *args: pbar.update()

    pool = mp.Pool(processes=mp.cpu_count() - 2)

    for f in range(permutation_num):
        coe=pool.apply_async(permutation_test, args=(xnew,reg,score,f), callback=update)
        X.append(coe)
    pool.close()
    pool.join()

    for res in X:
        result.append(res.get())
    result.append(coe2)
    result = np.array(result)
    # plot_his(result)

    print(result[permutation_num])
    print((np.sum(result[0:permutation_num]>=result[permutation_num]))/(permutation_num+1))

#save result
    savemat('D:/Desktop/result/bip_adhd/total2bd.mat',mdict={'Cor': result})
