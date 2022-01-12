from os import listdir
import numpy
from sklearn.model_selection import StratifiedKFold
from CR_WOA_DEF import CRWOA
from LIN_WOA_DEF import LINWOA
import scipy.io
from multiprocessing import Pool
from functools import partial
from KNNCV import KNNCrossValidation
from NNCV import NNCrossValidation
from NB import NBCrossValidation


def read_data(path):
    data=scipy.io.loadmat(path)
    train=data['train']
    label=data['label']
    return train,label

def PAR(reruns,P,T,path,M):
    p=Pool(2)
    if(M=='LINWOA'):
        PF=partial(PLIN,P=P,T=T,path=path)
    elif(M=='CRWOA'):
        PF=partial(PCR,P=P,T=T,path=path)
    output=p.map(PF,range(0,reruns))
    p.close()
    Cost=numpy.zeros(reruns,dtype=numpy.float64)
    CC=numpy.zeros([reruns,T],dtype=numpy.float64)
    Pos=[]
    for i in range(0,reruns):
        unpack=output[i]
        Cost[i]=unpack[0]
        Pos.append(unpack[1]) 
        CC[i,:]=unpack[2]
    return Cost,Pos,CC

def PLIN(re,P,T,path):
    train,label=read_data(path)
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=re)
    a,b,c=LINWOA(train,label,cv,P,T)
    return a,b,c

def PCR(re,P,T,path):
    train,label=read_data(path)
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=re)
    a,b,c=CRWOA(train,label,cv,P,T)
    return a,b,c

def knn(train,label,reruns):
    acc=numpy.zeros((reruns,10))
    for i in range(0,reruns):
        cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
        for k in range(1,11):
            acc[i,k-1]=KNNCrossValidation(train,label,cv,k)    
    ACC=numpy.max(acc,axis=1)
    return 1-ACC

def total_classes(label):
    unique=[]
    for i in label:
        if i not in unique:
            unique.append(i)
    
    total_class=len(unique)
    return total_class

def nn(train,label,reruns):
    dim=train.shape[1]
    ul=int(max(10,numpy.floor(numpy.sqrt(dim*total_classes(label)))))
    acc=numpy.zeros((reruns,ul))
    for i in range(0,reruns):
        cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
        for nn in range(1,ul):
            acc[i,nn-1]=NNCrossValidation(train,label,cv,nn)
    ACC=numpy.max(acc,axis=1)
    return 1-ACC

def nb(train,label,reruns):
    acc=numpy.zeros((reruns))
    for i in range(0,reruns):
        cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
        acc[i]=NBCrossValidation(train,label,cv)    
    return 1-acc

def main():
    path='./datasets'
    direc=sorted(listdir(path))
    print(direc)
    total_reruns=10
    Population=4
    Total_iter=50
    Cost=numpy.zeros([len(direc),total_reruns,5],dtype=numpy.float64)
    CC=numpy.zeros([len(direc),total_reruns,Total_iter,5],dtype=numpy.float64)
    Pos4=[]
    Pos5=[]
    for i in range(0,len(direc)):
        data_path=path+'/'+direc[i]
        print(data_path)
        train,label=read_data(data_path)
        Cost[i,:,0]=knn(train,label,total_reruns)
        Cost[i,:,1]=nn(train,label,total_reruns)
        Cost[i,:,2]=nb(train,label,total_reruns)
        Cost[i,:,3],Elite_pos,CC[i,:,:,3]=PAR(total_reruns,Population,Total_iter,data_path,'LINWOA')
        Pos4.append(Elite_pos)
        Cost[i,:,4],Elite_pos,CC[i,:,:,4]=PAR(total_reruns,Population,Total_iter,data_path,'CRWOA')
        Pos5.append(Elite_pos)
    return Cost,Pos4,Pos5,CC
    
Cost,Pos4,Pos5,CC = main()
