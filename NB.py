from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from numpy import mean

def NBTrain(train_data,train_label):
    model = GaussianNB(priors=None)
    model.fit(train_data,train_label)
    return model

def NBPredict(test_data,test_label,model):
    pred_label=model.predict(test_data)
    acc=accuracy_score(test_label,pred_label)
    return acc

def NBCrossValidation(train,label,cv):
    acc=[]
    dim=train.shape
    for train_index, test_index in cv.split(train,label):
        train_data=train[train_index,0:dim[1]]
        train_label=label[train_index]
        test_data=train[test_index,0:dim[1]]
        test_label=label[test_index]
        model=NBTrain(train_data,train_label)
        acc.append(NBPredict(test_data,test_label,model))
    accuracy=mean(acc)
    return accuracy