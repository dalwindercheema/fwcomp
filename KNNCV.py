from sklearn import neighbors
from sklearn.metrics import accuracy_score
from numpy import mean

def KNNTrain(train_data,train_label,K,weight):
    model = neighbors.KNeighborsClassifier(K, weights=weight)
    model.fit(train_data,train_label)
    return model

def KNNPredict(test_data,test_label,model):
    pred_label = model.predict(test_data)
    acc=accuracy_score(test_label,pred_label)
    return acc

def KNNCrossValidation(train,label,cv,K):
    acc=[]
    dim=train.shape
    for train_index, test_index in cv.split(train,label):
        train_data=train[train_index,0:dim[1]]
        train_label=label[train_index]
        test_data=train[test_index,0:dim[1]]
        test_label=label[test_index]
        model=KNNTrain(train_data,train_label,int(K),'uniform')
        acc.append(KNNPredict(test_data,test_label,model))
    accuracy=mean(acc)
    return accuracy