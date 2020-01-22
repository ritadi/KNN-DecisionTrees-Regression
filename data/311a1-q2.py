import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import graphviz

def load_data(fake_file, real_file):
    dataset = []
    f = open(fake_file)
    line = f.readline()
    while (line):
        dataset.append([line,0])
        line = f.readline()
    f.close()
    
    f = open(real_file)
    line = f.readline()
    while (line):
        dataset.append([line,1])
        line = f.readline()
    f.close()
    
    np.random.seed(0)
    np.random.shuffle(dataset)
    
    str_array = []
    label_array = []
    for i in range(0, len(dataset)-1):
        str_array.append(dataset[i][0])
        label_array.append(dataset[i][1])
        
    count_vectorizer = CountVectorizer(analyzer='word')
    dataset_vector = count_vectorizer.fit_transform(str_array).toarray()
    dataset_vector_feature_names = count_vectorizer.get_feature_names()
    
    headline_train, headline_temp, label_train, label_temp = train_test_split(dataset_vector, label_array, test_size = 0.3,shuffle=False)
    headline_vad, headline_test, label_vad, label_test = train_test_split(headline_temp, label_temp, test_size = 0.5,shuffle=False)

    return headline_train, headline_vad, headline_test, label_train, label_vad, label_test,dataset_vector_feature_names,dataset

    
def select_model(headline_train, label_train, headline_vad, label_vad):
    criteria = ['entropy', 'gini']
    maxdepth = [ 2, 4, 6, 8, 10]
    max_score = 0
    best_model = None
    for c in criteria:
        for d in maxdepth:
            model = DecisionTreeClassifier(criterion=c, max_depth=d)
            ##print(model)           
            model.fit(headline_train, label_train)
            result = model.predict(headline_vad)
            score = model.score(headline_vad, label_vad)
            print(score)
            if score > max_score:
                max_score = score
                best_model = model
    return best_model

        
def extract_and_visualize(best_model,features):
    classnames = ['fake', 'real']
    return export_graphviz(best_model,feature_names = features, class_names = classnames, 
                           max_depth=3,filled=True,rounded=True)
            

def compute_information_gain(keyword, dataset, headline_train):
    keyword_in_real = 0
    keyword_notin_real = 0
    keyword_in_fake = 0
    keyword_notin_fake = 0
    training_data = dataset[:len(headline_train)]    
    
    for i in range(0, len(headline_train)):
        if keyword in training_data[i][0] and training_data[i][1] == 1:
            keyword_in_real += 1
        elif keyword not in training_data[i][0] and training_data[i][1]  == 1:
            keyword_notin_real += 1                  
        elif keyword in training_data[i][0] and training_data[i][1]  == 0:
            keyword_in_fake += 1    
        elif keyword not in training_data[i][0] and training_data[i][1]  == 0:
            keyword_notin_fake += 1 

    real = float(keyword_in_real + keyword_notin_real)
    fake = float(keyword_in_fake + keyword_notin_fake)
    bothin = float(keyword_in_real + keyword_in_fake)
    bothout = float(keyword_notin_real + keyword_notin_fake)
    
    H_keyword = -(real/len(headline_train)) * np.log2(real/len(headline_train)) - (fake/len(headline_train)) * np.log2(fake/len(headline_train))
    
    H_Cond_keyword = (- (keyword_in_real/bothin) * np.log2(keyword_in_real/bothin) - (keyword_in_fake/bothin) * np.log2(keyword_in_fake/bothin))*(bothin/len(headline_train)) +(- (keyword_notin_real/bothout) * np.log2(keyword_notin_real/bothout) - (keyword_notin_fake/bothout) * (np.log2(keyword_notin_fake/bothout)))*(bothout/len(headline_train))
            
    result = round(H_keyword - H_Cond_keyword, 4)
    print("The information gain for '", keyword,"' is: ", result);
    
    return result
    
    
    
            
if __name__ == "__main__":
    headline_train, headline_vad, headline_test, label_train, label_vad, label_test, features, dataset = load_data("./clean_fake.txt", "./clean_real.txt")
    
    best_model = select_model(headline_train,label_train,headline_vad,label_vad)
    
    print(best_model)
    tree = extract_and_visualize(best_model, features)
    print(tree)


    compute_information_gain("the", dataset, headline_train)
    compute_information_gain("clinton", dataset, headline_train)
    compute_information_gain("hillary", dataset, headline_train)
