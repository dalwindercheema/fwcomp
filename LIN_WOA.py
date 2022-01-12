import numpy
import math
from LINEARSVMCV import LLinearSVMCrossValidation
from Initialization import initialization_NSVML
from weights import calc_weights

def LINWOA(train,label,cv,P,Iter):
    size=train.shape
    dim=size[1]
    lb=numpy.zeros([dim,1])
    ub=numpy.ones([dim,1])
    lb=numpy.append(lb,0.01)
    ub=numpy.append(ub,35000)
    positions=initialization_NSVML(lb,ub,P)
    Leader_score=1
    Leader_pos=[]
    CC=[]
    curr_iter=0
    while (curr_iter<Iter):
#        print(curr_iter)
        for i in range(0,P):
            for j in range(0,dim+1):
                positions[i,j]=max(min(positions[i,j],ub[j]),lb[j])    
            
            weighted_train=calc_weights(train,positions[i,0:dim])
            sze=weighted_train.shape
            if(sze[1]==0):
                fitness=1
            else:
                fitness=1-LLinearSVMCrossValidation(weighted_train,label,cv,positions[i,dim])
                
            if fitness<Leader_score:
                Leader_score=fitness
                Leader_pos=numpy.copy(positions[i,:])
        
        a=4-curr_iter*((2)/Iter)
        a2=-1+curr_iter*((-1)/Iter)
        for i in range(0,P):
            r1=numpy.random.rand()
            r2=numpy.random.rand()
            
            A=2*a*r1-a
            C=2*r2
            b=1
            l=(a2-1)*numpy.random.rand()+1
            p=numpy.random.rand()
            
            for j in range(0,dim+1):
                if p<0.5:
                    if(abs(A)>=1):
                        rand_index = math.floor(P*numpy.random.rand())
                        X_rand = positions[rand_index,:]
                        D_X_rand=abs(C*X_rand[j]-positions[i,j])
                        positions[i,j]=X_rand[j]-A*D_X_rand
                    else:
                        D_Leader=abs(C*Leader_pos[j]-positions[i,j])
                        positions[i,j]=Leader_pos[j]-A*D_Leader
                else:
                    distance2Leader=abs(Leader_pos[j]-positions[i,j])
                    positions[i,j]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j]
        curr_iter=curr_iter+1
        CC.append(Leader_score)
    return Leader_score,Leader_pos,CC
