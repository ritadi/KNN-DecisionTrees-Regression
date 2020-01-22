import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    length = np.arange(len(data["X"]))
    np.random.seed(0)
    index = np.random.permutation(length)
    
    result = {"X":[], "t":[]}    
    for i in index:
        result["X"].extend(data["X"][index])
        result["t"].extend(data["t"][index])

    return result


def split_data(data, num_folds, fold):
    fold_size = int(len(data["X"])//num_folds)
    data_fold = {"X":[],"t":[]}
    data_rest = {"X":[], "t":[]}

    data_fold["X"] = data["X"][(fold-1)*fold_size: fold*fold_size]
    data_fold["t"] = data["t"][(fold-1)*fold_size: fold*fold_size]
    data_rest["X"] = data["X"][0: (fold-1)*fold_size] + data["X"][fold*fold_size: len(data["X"])]
    data_rest["t"] = data["t"][0: (fold-1)*fold_size] + data["t"][fold*fold_size: len(data["t"])]
    return data_fold, data_rest


def train_model(data, lambd):
    train_data = np.array(data["X"])
    label = np.array(data["t"]) 
    label.reshape(len(data["t"]), 1)
    I_shape = train_data.shape[1]
    data_transpose = train_data.transpose()
    dataT_data = np.dot(data_transpose, train_data)
    inv = np.linalg.inv(dataT_data+lambd*np.identity(I_shape))
    inv_data_transpose = np.dot(inv, data_transpose)
    
    result = np.dot(inv_data_transpose, label)
    return result
    
def predict(data, model):
    train_data = np.array(data["X"])
    w = np.array(model).reshape(train_data.shape[1],1)
    prediction = np.dot(train_data, w)
    
    return prediction
    
def loss(data, model):
    prediction = np.array(predict(data, model)).reshape(len(data["t"]) ,1)
    target = np.array(data["t"]).reshape(len(data["t"]) ,1)
    sum_sqr = 0
    for i in range(0, len(prediction)):
        sum_sqr += (target[i] - prediction[i]) ** 2
    
    return sum_sqr/len(data["t"])


def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = len(lambd_seq) * [0]
    for i in range(0, len(lambd_seq)):
        print("cross", i)
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds+1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = cv_loss_lmd/num_folds
    return cv_error
    
def lambd_seq_error(train_data, test_data, lambd_seq):
    
    train_error = []
    test_error = []
    for i in range(0, len(lambd_seq)):
        
        model = train_model(train_data, lambd_seq[i])
        train_error.append(loss(train_data, model))
        test_error.append(loss(test_data, model))
    return train_error, test_error
    
    
    
if __name__ == "__main__":
    data_train = {"X": np.genfromtxt("data_train_X.csv", delimiter=","),
                  "t": np.genfromtxt("data_train_y.csv", delimiter=",")}
    data_test = {"X": np.genfromtxt("data_test_X.csv", delimiter=","),
                 "t": np.genfromtxt("data_test_y.csv", delimiter=",")}
    lambd_seq = []
    acc = 0.02
    for i in range(0, 50):
        acc += (1.5-0.02)/50
        lambd_seq.append(acc)
    
    training_err, test_err = lambd_seq_error(data_train, data_test, lambd_seq)
    for i in range(0, len(lambd_seq)):
        print("training error of lambd " ,lambd_seq[i]," is : ",training_err[i][0])
    for i in range(0, len(lambd_seq)):
        print("test error of lambd " ,lambd_seq[i]," is : ",test_err[i][0])        
        
    f_fold = cross_validation(data_train, 5, lambd_seq)
    t_fold = cross_validation(data_test, 10, lambd_seq)
    
    plt.plot(lambd_seq, training_err, 'mo-') 
    plt.plot(lambd_seq, test_err, 'bo-') 
    plt.plot(lambd_seq, f_fold, 'go-') 
    plt.plot(lambd_seq, t_fold, 'yo-')  
    
    plt.title('lambd tuning')
    plt.ylabel('Error')
    plt.xlabel('lambd')
    plt.legend(['Training Error', 'Test Error','f_fold','t_fold'], loc='upper right')    
    plt.show()
    
    