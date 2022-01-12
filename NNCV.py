from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from numpy import mean

def NNTrain(train_data,train_label,neurons):
    model = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(neurons,),activation='tanh')
    model.fit(train_data,train_label)
    return model

def NNPredict(test_data,test_label,model):
    pred_label = model.predict(test_data)
    acc=accuracy_score(test_label,pred_label)
    return acc

def NNCrossValidation(train,label,cv,neurons):
    acc=[]
    dim=train.shape
    for train_index, test_index in cv.split(train,label):
        train_data=train[train_index,0:dim[1]]
        train_label=label[train_index]
        test_data=train[test_index,0:dim[1]]
        test_label=label[test_index]
        model=NNTrain(train_data,train_label,int(neurons))
        acc.append(NNPredict(test_data,test_label,model))
    accuracy=mean(acc)
    return accuracy