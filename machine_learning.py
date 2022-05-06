
import numpy as fn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFpr
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

inputfile='D:/Desktop/study/bip/manuscript_bip/data/Hse.txt'
brain=fn.loadtxt(inputfile)
inputfile='D:/Desktop/study/bip/manuscript_bip/data/bip_adhd.txt'
bd=fn.loadtxt(inputfile)
# inputfile='D:/Desktop/study/bip/manuscript_bip/data//bipolar.txt'
# other=fn.loadtxt(inputfile)


reg = linear_model.LinearRegression()
F,pval=f_regression(brain,bd)
print(min(pval))
fn.savetxt('D:/Desktop/Hse_Fvalue.dat', F)

coe1=[];
for num in range(1,len(F),1):
 xnew=SelectKBest(f_regression, k=num).fit_transform(brain,bd)
 scores = cross_val_predict(reg,xnew,bd, cv=49)
 cor=np.corrcoef(bd,scores)
 coe1.append(cor[0,1])
print('age',max(coe1),coe1.index(max(coe1)))

coe3=[];
xnew=SelectKBest(f_regression, k=coe1.index(max(coe1)) + 1).fit_transform(brain,bd)
scores = cross_val_predict(reg,xnew,bd, cv=49)
cor=np.corrcoef(bd,scores)
coe3.append(cor[0,1])
print('age',max(coe3))
fn.savetxt('D:/Desktop/Hse_prediction.dat',scores)

###=====================weight
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
coe2=[];
for train_index, test_index in loo.split(xnew):
#print("TRAIN:", train_index, "TEST:", test_index)
 X_train, Y_train = xnew[train_index], bd[train_index]
 reg.fit(X_train, Y_train)
 coe2.append(reg.coef_)
fn.savetxt('D:/Desktop/Hse_coee.dat', coe2)
#
# fig, ax = plt.subplots()
# ax.scatter(scores,bd, edgecolors=(0, 0, 0))
# ax.set_xlabel('predicted')
# ax.set_ylabel('score')
# plt.show()