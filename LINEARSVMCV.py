from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from numpy import mean
from warnings import filterwarnings

filterwarnings('ignore')

def LinearSVMTrain(train_data,train_label,C_value):
    model = LinearSVC(C=C_value)
    model.fit(train_data,train_label)
    model.sparsify()
    return model

def LinearSVMPredict(test_data,test_label,model):
    pred_label = model.predict(test_data)
    acc=accuracy_score(test_label,pred_label)
    return acc

def LLinearSVMCrossValidation(train,label,cv,C_value):
    acc=[]
    dim=train.shape
    for train_index, test_index in cv.split(train,label):
        train_data=train[train_index,0:dim[1]]
        train_label=label[train_index]
        test_data=train[test_index,0:dim[1]]
        test_label=label[test_index]
        model=LinearSVMTrain(train_data,train_label,C_value)
        acc.append(LinearSVMPredict(test_data,test_label,model))
    accuracy=mean(acc)
    return accuracy