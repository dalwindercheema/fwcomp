from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from numpy import mean


def RBFSVMTrain(train_data,train_label,C_value,gamma_value):
    model = SVC(kernel='rbf',C=C_value,gamma=gamma_value)
    model.fit(train_data,train_label)
    return model

def RBFSVMPredict(test_data,test_label,model):
    pred_label = model.predict(test_data)
    acc=accuracy_score(test_label,pred_label)
    return acc

def RBFSVMCrossValidation(train,label,cv,C_value,gamma_value):
    acc=[]
    dim=train.shape
    for train_index, test_index in cv.split(train,label):
        train_data=train[train_index,0:dim[1]]
        train_label=label[train_index]
        test_data=train[test_index,0:dim[1]]
        test_label=label[test_index]
        model=RBFSVMTrain(train_data,train_label,C_value,gamma_value)
        acc.append(RBFSVMPredict(test_data,test_label,model))
    accuracy=mean(acc)
    return accuracy