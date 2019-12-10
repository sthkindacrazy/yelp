import numpy as np
import pandas as pd
import data_loader as dl
import seaborn as sns
import matplotlib.pyplot as plt
import cleaning as cl
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def cat_combine(data):
    cln_text =[]
    cln_text_2 =[]
    for sentence in data["categories"]:
        cln_text.append(re.sub("['() [\]]", "", sentence))
    for sentence in cln_text:
        cln_text_2.append(re.sub("[,]"," ",sentence))
    data["cat_vec"]=cln_text_2
    return data

def get_final_pred(prob_mat_all):
    for i in range(len(prob_mat_all)):
        if i == 0:
            final_prob = prob_mat_all[i]
        else:
            final_prob = np.multiply(final_prob,prob_mat_all[i])

    pred_label = np.argmax(final_prob, axis=1)+1
    accuracy = round(accuracy_score(np.array(y_test), pred_label) * 100, 2)
    return accuracy


####one-hot encoding method
def _main_onehot_():
    #preppare_data():
    train_data = dl.load_clean_data('train')
    x_tem, y_tem = (train_data[0], train_data[-1])
    x_train_cl = cl.rm_stopwords(x_tem)    # use x_train_cl["cln_text"] to call the cleaned review
    x_train_cl = cat_combine(x_train_cl)
    x_train, x_test,y_train, y_test = train_test_split(x_train_cl,y_tem,test_size=0.2,random_state=42,shuffle=True)

    ###one hot encoding
    train_data_text, test_data_text = cl.phrase_one_hot_encode(x_train["cln_split_text"],x_test["cln_split_text"],35000)
    train_data_name, test_data_name = cl.phrase_one_hot_encode(x_train["name"],x_test["name"])
    train_data_cat, test_data_cat = cl.phrase_one_hot_encode(x_train["cat_vec"],x_test["cat_vec"])
    ##combine other feature
    train_use_cool = x_train.to_numpy()[:,3:6]
    test_use_cool = x_test.to_numpy()[:,3:6]
    train_n_sscore = x_train.to_numpy()[:,10:13]
    test_n_sscore = x_test.to_numpy()[:,10:13]
    train_factor=np.hstack((train_use_cool,train_n_sscore))
    test_factor=np.hstack((test_use_cool,test_n_sscore))

    prob_mat_all=[]
    ###text:
    softmax_text=LogisticRegression(multi_class="multinomial", solver = "newton-cg")
    softmax_text.fit(train_data_text,y_train)
    test_pred = softmax_text.predict(test_data_text)
    score_text = round(accuracy_score(y_test,test_pred) * 100, 2)
    text_prob_matrix=softmax_text.predict_proba(test_data_text)
    prob_mat_all.append(text_prob_matrix)
    ###cat:
    softmax_cat=LogisticRegression(multi_class="multinomial", solver = "newton-cg")
    softmax_cat.fit(train_data_cat,y_train)
    test_pred = softmax_cat.predict(test_data_cat)
    score_cat = round(accuracy_score(y_test, test_pred) * 100, 2)
    cat_prob_matrix=softmax_cat.predict_proba(test_data_cat)
    prob_mat_all.append(cat_prob_matrix)
    ####factor:
    softmax_factor=LogisticRegression(multi_class="multinomial", solver = "newton-cg")
    softmax_factor.fit(train_factor,np.array(y_train))
    test_pred = softmax_factor.predict(test_factor)
    score_factor = round(accuracy_score(np.array(y_test), test_pred) * 100, 2)
    factor_prob_matrix=softmax_factor.predict_proba(test_factor)
    prob_mat_all.append(factor_prob_matrix)
    acc = get_final_pred(prob_mat_all)

    print("acc of one-hot encoding model is : ",acc)
    return


####topfreq method
def vocabulization(data):
    vocab=[]
    for str in data["cln_split_text"]:
        vocab.append(re.findall(r"[\w']+|[!?]", str))
    data['split_text'] = vocab
    return data
def get_top_words(n,data,y_train):
    words_by_rating={}
    words_by_rating['1 star']={}
    words_by_rating['2 star']={}
    words_by_rating['3 star']={}
    words_by_rating['4 star']={}
    words_by_rating['5 star']={}
    review = np.array(data["split_text"])
    rate = np.array(y_train)
    for stars in range(1,6,1):
        for sentence in review[rate==stars]:
            for word in sentence:
                if word in words_by_rating['%s star'%stars]:
                    words_by_rating['%s star'%stars][word] += 1
                else:
                    words_by_rating['%s star'%stars][word] = 1
    all_words={}
    for sentence in review:
        for word in sentence:
            if word in all_words:
                all_words[word] += 1
            else:
                all_words[word] = 1
    top_n_1_star_words=sorted(words_by_rating['1 star'],key=words_by_rating['1 star'].get,reverse=True)[:n]
    top_n_2_star_words=sorted(words_by_rating['2 star'],key=words_by_rating['2 star'].get,reverse=True)[:n]
    top_n_3_star_words=sorted(words_by_rating['3 star'],key=words_by_rating['3 star'].get,reverse=True)[:n]
    top_n_4_star_words=sorted(words_by_rating['4 star'],key=words_by_rating['4 star'].get,reverse=True)[:n]
    top_n_5_star_words=sorted(words_by_rating['5 star'],key=words_by_rating['5 star'].get,reverse=True)[:n]
    top_words_in_all =sorted(all_words,key=all_words.get,reverse=True)[:n]
    top_words_bag = {}
    top_words_bag["1 star"] = top_n_1_star_words
    top_words_bag["2 star"] = top_n_2_star_words
    top_words_bag["3 star"] = top_n_3_star_words
    top_words_bag["4 star"] = top_n_4_star_words
    top_words_bag["5 star"] = top_n_5_star_words

    return (top_words_bag,top_words_in_all)
def top_word_freq(n_star_words,data):
    review_pd = data["split_text"]
    review = np.array(review_pd)
    feat_matrix = []
    for j in range(len(review)):
        num = list(np.zeros([len(n_star_words)]))
        for i in range(len(n_star_words)):
            num[i] = review[j].count(n_star_words[i])
        feat_matrix.append(num)

    return feat_matrix

def star_matrix(top_words_bag,data):
    star_matrix = {}
    for star in range(1,6,1):
        feat_matrix = top_word_freq(top_words_bag['%s star'%star],data)
        star_matrix['%s star'%star] = feat_matrix
    return star_matrix

def form_one_bag(top_bag):
    one_bag_word_ref = np.array([])
    one_bag_word = np.array([])
    for star in range(1,6,1):
        for i in range(len(top_bag["1 star"])):
            if top_bag['%s star'%star][i] not in one_bag_word:
                one_bag_word_ref = np.append(one_bag_word,top_bag['%s star'%star])
    return one_bag_word_ref


###build the cat_based_top_frequent_words model

def top_freq_main_():
    #preppare_data():
    train_data = dl.load_clean_data('train')
    x_tem, y_tem = (train_data[0], train_data[-1])
    x_train_cl = cl.rm_stopwords(x_tem)    # use x_train_cl["cln_text"] to call the cleaned review
    x_train_cl = cat_combine(x_train_cl)
    x_train, x_test,y_train, y_test = train_test_split(x_train_cl,y_tem,test_size=0.2,random_state=42,shuffle=True)
    x_train = vocabulization(x_train)
    x_test = vocabulization(x_test)
    top_cat, top_all = get_top_words(2000,x_train,y_train)
    train_feat_mat=star_matrix(top_cat,x_train)
    test_feat_mat=star_matrix(top_cat,x_test)

    #build the logistic model
    softmax_reg={}
    for star in range(1,6,1):
        x= np.array(train_feat_mat['%s star'%star])
        y = np.array(y_train)
        softmax_reg['%s star'%star]=LogisticRegression(multi_class="multinomial", solver = "newton-cg")
        softmax_reg['%s star'%star].fit(x,y)
    for star in range(1,6,1):
        x= np.array(test_feat_mat['%s star'%star])
        if star == 1:
            prob_mat = softmax_reg['%s star'%star].predict_proba(x)
        else:
            prob_mat = np.multiply(prob_mat,softmax_reg['%s star'%star].predict_proba(x))

    ##combine other feature
    train_use_cool = x_train.to_numpy()[:,3:6]
    test_use_cool = x_test.to_numpy()[:,3:6]
    train_n_sscore = x_train.to_numpy()[:,10:13]
    test_n_sscore = x_test.to_numpy()[:,10:13]
    train_factor=np.hstack((train_use_cool,train_n_sscore))
    test_factor=np.hstack((test_use_cool,test_n_sscore))

    prob_mat_all=[]

    ###cat:
    softmax_cat=LogisticRegression(multi_class="multinomial", solver = "newton-cg")
    softmax_cat.fit(train_data_cat,y_train)
    test_pred = softmax_cat.predict(test_data_cat)
    score_cat = round(accuracy_score(y_test, test_pred) * 100, 2)
    cat_prob_matrix=softmax_cat.predict_proba(test_data_cat)
    prob_mat_all.append(cat_prob_matrix)
    ####factor:
    softmax_factor=LogisticRegression(multi_class="multinomial", solver = "newton-cg")
    softmax_factor.fit(train_factor,np.array(y_train))
    test_pred = softmax_factor.predict(test_factor)
    score_factor = round(accuracy_score(np.array(y_test), test_pred) * 100, 2)
    factor_prob_matrix=softmax_factor.predict_proba(test_factor)
    prob_mat_all.append(factor_prob_matrix)


    prob_mat_all.append(cat_prob_matrix)
    prob_mat_all.append(factor_prob_matrix)
    prob_mat_all.append(prob_mat)
    acc = get_final_pred(prob_mat_all)

    print("acc of top_freq_words based model is: ",acc)
    return


# In[ ]:

## call this two function to get results
top_freq_main_()
_main_onehot_()
