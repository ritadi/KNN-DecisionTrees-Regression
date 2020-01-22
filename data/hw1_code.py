
import numpy as np
from sklearn.feature_extraction import text
from sklearn import tree
import graphviz

def load_data(data_file1, data_file2):
    file_list=[data_file1, data_file2]
    data = []
    for i in file_list:
        f=open(i,"r")
        f.readline()
        if "fake" in i:
            for line in f:
                data.append([line, 0])
        else:
            for line in f:
                data.append([line, 1])
		
    np.random.seed(100)
    np.random.shuffle(data)
    news_title_array =[]
	
    for i in range(0, len(data) - 1):
        news_title_array.append(data[i][0])
	
    dataset_target = []
    for i in range(0, len(data) - 1):
        dataset_target.append(data[i][1])

    vectorizer = text.TfidfVectorizer(analyzer='word')
    dataset_X = vectorizer.fit_transform(news_title_array).toarray()
    dataset_X_feature_names = vectorizer.get_feature_names()

    # splits the entire data set randomly into 70% training, 15% validation, and 15% test examples
    training_cutoff = int(round(len(dataset_X) * 0.7))
    validation_cutoff = int(round(len(dataset_X) * 0.85))
    test_cutoff = len(dataset_X) - 1

    training_X = dataset_X[0:int(training_cutoff)]
    training_target = dataset_target[0:int(training_cutoff)]
    #print(dataset_X_feature_names)

    # print(training_target)

    validation_X = dataset_X[training_cutoff + 1:validation_cutoff]
    validdation_target = dataset_target[training_cutoff + 1:validation_cutoff]
    
    test_X = dataset_X[validation_cutoff + 1:test_cutoff]
    test_target = dataset_target[validation_cutoff + 1:test_cutoff]
    
    return dataset_X_feature_names, training_X, validation_X, test_X, training_target, validdation_target, test_target,data




def select_model(feature_names, training_X, validation_X, t_target, v_target ):
    for criterion in ['entropy', 'gini']:
        for i in range(4, 15, 2):
            clf = tree.DecisionTreeClassifier(max_depth=i, criterion=criterion)
            clf = clf.fit(training_X, t_target)
            
#	    print("criterion=" + criterion + ", max depth = " + str(i) + ", score is " + str(clf.score(validation_X, v_target)))
            # draw graph
	    dot_data = tree.export_graphviz(clf, out_file=None,
                                            feature_names=feature_names,
                                            class_names=['fake', 'real'],
                                            filled=True, rounded=True,
                                            special_characters=True)
	#graph = graphviz.Source(dot_data)
	#name= "news:" + "criterion=" + criterion + ", max depth = " + str(i)+".jpg"
	#graph.render(name,view=True)


def compute_information_gain(xi,data):
    
    train_cutoff = int(round(len(data)*0.70))
    
    training = data[:train_cutoff]
    
    real_t1=0
    real_t2=0
    fake_t1=0
    fake_t2=0
    for i in range(train_cutoff):
        #real head news
        if data[i][1]==1:
            if xi in training[i][0].split(" "):
                real_t1+=1
            else:
                real_t2+=1
        #false head news
        elif data[i][1]==0:
            if xi in training[i][0].split(" "):
                fake_t1+=1
            else:
                fake_t2+=1

    real_t=float(real_t1+real_t2)
    fake_t=float(fake_t1+fake_t2)
    x1=float(real_t1+fake_t1)
    x2=float(real_t2+fake_t2)
    
    
    
    H_Y= -(real_t/train_cutoff) * np.log2(real_t/train_cutoff) - (fake_t/train_cutoff) * np.log2(fake_t/train_cutoff)
    H_X_Y= (- (real_t1/x1) * np.log2(real_t1/x1) - (fake_t1/x1) * np.log2(fake_t1/x1))*(x1/train_cutoff) +(- (real_t2/x2) * np.log2(real_t2/x2) - (fake_t2/x2) * (np.log2(fake_t2/x2)))*(x2/train_cutoff)
    information_gain=H_Y-H_X_Y
    print("keyword: ",xi,";information gain: ",round(information_gain,4))
    
    return round(information_gain,4)
	
	
if __name__ == "__main__":
    dataset_X_feature_names, training_X, validation_X, test_X, training_target, validdation_target, test_target,data=load_data("./clean_fake.txt","./clean_real.txt")
    #select_model(dataset_X_feature_names, training_X, validation_X, training_target, validdation_target)
    compute_information_gain("american",data)



