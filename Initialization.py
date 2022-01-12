import numpy

def initialization_KNN(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            if(j==dim[0]-1):
                pos[i,j]=numpy.rint(numpy.random.uniform(lb[j],ub[j],1))
            else:
                pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_NN(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            if(j==dim[0]-1):
                pos[i,j]=numpy.rint(numpy.random.uniform(lb[j],ub[j],1))
            else:
                pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_NB(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_CSVML(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_CSVMP(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_CSVMR(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_NSVML(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_NSVMP(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_NSVMR(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos

def initialization_def(lb,ub,P):
    dim=lb.shape
    pos=numpy.zeros([P,dim[0]]);
    for i in range(0,P):
        for j in range(0,dim[0]):
            pos[i,j]=numpy.random.uniform(lb[j],ub[j],1)          
    return pos
