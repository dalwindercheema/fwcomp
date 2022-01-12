from os import listdir
import numpy
from sklearn.model_selection import StratifiedKFold
from KNN_WOA import KNNWOA
from NN_WOA import NNWOA
from NB_WOA import NBWOA
from CR_WOA import CRWOA
from LIN_WOA import LINWOA
import scipy.io
from multiprocessing import Pool
from functools import partial

def read_data(path):
    data=scipy.io.loadmat(path)
    train=data['train']
    label=data['label']
    return train,label

def PAR(reruns,P,T,path,M):
    p=Pool(10)
    if(M=='KNNWOA'):
        PF=partial(PKNN,P=P,T=T,path=path)
    elif(M=='NNWOA'):
        PF=partial(PNN,P=P,T=T,path=path)
    elif(M=='NBWOA'):
        PF=partial(PNB,P=P,T=T,path=path)
    elif(M=='LINWOA'):
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

def PKNN(re,P,T,path):
    train,label=read_data(path)
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=re)
    a,b,c=KNNWOA(train,label,cv,P,T)
    #print('x')
    return a,b,c

def PNN(re,P,T,path):
    train,label=read_data(path)
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=re)
    a,b,c=NNWOA(train,label,cv,P,T)
    return a,b,c

def PNB(re,P,T,path):
    train,label=read_data(path)
    cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=re)
    a,b,c=NBWOA(train,label,cv,P,T)
    return a,b,c

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

def main():
    path='./datasets'
    direc=sorted(listdir(path))
    print(direc)
    total_reruns=10
    Population=8
    Total_iter=100
    Cost=numpy.zeros([len(direc),total_reruns,5],dtype=numpy.float64)
    CC=numpy.zeros([len(direc),total_reruns,Total_iter,5],dtype=numpy.float64)
    Pos1=[]
    Pos2=[]
    Pos3=[]
    Pos4=[]
    Pos5=[]
    for i in range(0,len(direc)):
        data_path=path+'/'+direc[i]
        print(data_path)
        Cost[i,:,0],Elite_pos,CC[i,:,:,0]=PAR(total_reruns,Population,Total_iter,data_path,'KNNWOA')
        Pos1.append(Elite_pos)
        Cost[i,:,1],Elite_pos,CC[i,:,:,1]=PAR(total_reruns,Population,Total_iter,data_path,'NNWOA')
        Pos2.append(Elite_pos)
        Cost[i,:,2],Elite_pos,CC[i,:,:,2]=PAR(total_reruns,Population,Total_iter,data_path,'NBWOA')
        Pos3.append(Elite_pos)
        Cost[i,:,3],Elite_pos,CC[i,:,:,3]=PAR(total_reruns,Population,Total_iter,data_path,'LINWOA')
        Pos4.append(Elite_pos)
        Cost[i,:,4],Elite_pos,CC[i,:,:,4]=PAR(total_reruns,Population,Total_iter,data_path,'CRWOA')
        Pos5.append(Elite_pos)
    return Cost,Pos1,Pos2,Pos3,Pos4,Pos5,CC
    
Cost,Pos1,Pos2,Pos3,Pos4,Pos5,CC = main()
