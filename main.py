import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import data_loader as dl
import cleaning as cl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import mglearn

# load the data in just the same procesude as before
train_data = dl.load_clean_data('train')
x_tem, y_tem = (train_data[0], train_data[-1]) 

#remove all the punctuations and stop words (special repeated char like "``" still remains for now)
x_train_cl = cl.rm_stopwords(x_tem)    # use x_train_cl["cln_text"] to call the cleaned review
#x_train, x_test,y_train, y_test = train_test_split(x_train_cl,y_tem,test_size=0.2,random_state=42,shuffle=True) # we can swtich "42" to "RandomState" later

#####
#####Just keep this part of the code above same for all three of us to maintain an identical initialization of review text

def multinomial_bayes():
    x_train_ref_org, x_test_ref_org, y_train_ref, y_test_ref = train_test_split(x_train_cl,y_tem,test_size=0.2,random_state=42,shuffle=True)
    num_feature = []
    acc = []
    for i in range(1000, 10000, 1000):
        mnb = MultinomialNB()
        x_train_ref = x_train_ref_org
        x_test_ref = x_test_ref_org
        #x_train_ref = cl.phrase_tf_idf_encode(x_train_ref, i)
        #x_test_ref = cl.phrase_tf_idf_encode(x_test_ref, i)
        x_train_ref, x_test_ref = cl.phrase_one_hot_encode(x_train_ref, x_test_ref, i)
        mnb.fit(x_train_ref, y_train_ref)
        predmnb = mnb.predict(x_test_ref)
        score = round(accuracy_score(y_test_ref, predmnb) * 100, 2)
        print(i, score)
        num_feature.append(i)
        acc.append(score)
    plt.plot(num_feature, acc)
    plt.xlabel('number of words')
    plt.ylabel('classification accuracy')
    plt.show()


def logistic():
    x_train_ref_org, x_test_ref_org, y_train_ref, y_test_ref = train_test_split(x_train, y_train, test_size=0.2,
                                                                                random_state=42)
    vectorizer = CountVectorizer(max_features=10000, min_df=1, ngram_range=(1, 2))
    vectorizer.fit(x_train_ref_org['text'])

    x_train_ref =vectorizer.transform(x_train_ref_org['text'])
    x_test_ref = vectorizer.transform(x_test_ref_org['text'])

    feature_names = vectorizer.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(x_train_ref, y_train_ref)

    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)

    lr = grid.best_estimator_
    lr.predict(x_test_ref)
    print("Score: {:.2f}".format(lr.score(x_test_ref, y_test_ref)))







