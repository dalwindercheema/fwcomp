import numpy

def calc_weights(train,feat_weights):
    dim=train.shape[1]
    total_feat=numpy.count_nonzero(feat_weights)
    ntrain=numpy.zeros((train.shape[0],total_feat))
    cont=0
    for i in range(0,dim):
        if(feat_weights[i]!=0):
            ntrain[:,cont]=train[:,i]*feat_weights[i]
            cont=cont+1
    return ntrain