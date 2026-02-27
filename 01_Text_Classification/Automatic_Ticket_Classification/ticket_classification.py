import json 
import numpy as np
import pandas as pd
import re, nltk, spacy, string
from wordcloud import WordCloud
# spacy.require_gpu()
import en_core_web_sm
nlp = en_core_web_sm.load()
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import pdb
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from pprint import pprint
from tqdm import tqdm, tqdm_notebook
from sklearn.decomposition import NMF
tqdm.pandas()

# importing libraries required for model building and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split

from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
stopwords = nlp.Defaults.stop_words

def load_data(data_num=10000):
    f = open('data/complaints-2021-05-14_08_16.json')
    data = json.load(f)
    df=pd.json_normalize(data)
    df = df[:data_num]
    return df

def clean_text(text):
  text=text.lower()  #convert to lower case
  text=re.sub(r'^\[[\w\s]\]+$',' ',text) #Remove text in square brackets
  text=re.sub(r'[^\w\s]',' ',text) #Remove punctuation
  text=re.sub(r'^[a-zA-Z]\d+\w*$',' ',text) #Remove words with numbers
  return text

def lemmatization(texts):
    lemma_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.lemma_ for token in doc if token.text not in set(stopwords)]
        lemma_sentences.append(' '.join(sent))
    return lemma_sentences

def extract_pos_tags(texts):
    pos_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.text for token in doc if token.tag_ == 'NN']
        pos_sentences.append(' '.join(sent))
    return pos_sentences

def preprocess_data(data_df):
    data_df.rename(columns={'_index':'index',
    '_type':'type',
    '_id':'id',
    '_score':'score',
    '_source.tags':'tags',
    '_source.zip_code':'',
    '_source.complaint_id':'complaint_id',
    '_source.issue':'issue',
    '_source.date_received':'date_received',
    '_source.state':'state',
    '_source.consumer_disputed':'consumer_disputed',
    '_source.product':'product',
    '_source.company_response':'company_response',
    '_source.company':'company',
    '_source.submitted_via':'submitted_via',
    '_source.date_sent_to_company':'date_sent_to_company',
    '_source.company_public_response':'company_public_response',
    '_source.sub_product':'sub_product',
    '_source.timely':'timely',
    '_source.complaint_what_happened':'complaint_what_happened',
    '_source.sub_issue':'sub_issue',
    '_source.consumer_consent_provided':'consumer_consent_provided'},inplace=True)

    data_df['complaint_what_happened'].replace('', np.nan, inplace=True)
    print(data_df['complaint_what_happened'].isnull().sum())
    data_df.dropna(subset=['complaint_what_happened'],inplace=True)
    
    # clean data
    df_clean = pd.DataFrame()
    
    # clean text, lemmantize
    df_clean['complaint_what_happened'] = data_df['complaint_what_happened'].progress_apply(lambda x: clean_text(x))
    df_clean['complaint_what_happened_lemmatized'] = lemmatization(df_clean['complaint_what_happened'])
    df_clean['category'] = data_df['product']
    df_clean['sub_category'] = data_df['sub_product']
    
    # Extract POS
    df_clean["complaint_POS_removed"] = extract_pos_tags(df_clean['complaint_what_happened_lemmatized'])
    return df_clean

def get_top_n_words(corpus, n=None,count=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:count]

def eda(data_df):
    data_df['complaint_length'] = data_df['complaint_what_happened'].str.len()
    data_df['complaint_what_happened_lemmatized_length'] = data_df['complaint_what_happened_lemmatized'].str.len()
    data_df['complaint_POS_removed_length'] = data_df['complaint_POS_removed'].str.len()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_df['complaint_length'], name='Complaint'))
    fig.add_trace(go.Histogram(x=data_df['complaint_what_happened_lemmatized_length'], name='Complaint Lemmatized'))
    fig.add_trace(go.Histogram(x=data_df['complaint_POS_removed_length'], name='Complaint POS Removed'))
    fig.update_layout(barmode='overlay', title='Complaint Character Length', xaxis_title='Character Length', yaxis_title='Count')
    fig.update_traces(opacity=0.75)
    fig.write_image("eda_complaint_character_length.png")
    
    
    #Using a word cloud find the top 40 words by frequency among all the articles after processing the text

    wordcloud=WordCloud(stopwords=stopwords, background_color='white', width=2000, height=1500,max_words=40).generate(' '.join(data_df['complaint_POS_removed']))
    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')
    plt.axis("off")
    plt.savefig("eda_wordcloud.png")

    data_df['Complaint_clean'] = data_df['complaint_POS_removed'].str.replace('-PRON-', '')

    #Print the top 10 words in the unigram frequency and plot the same using a bar graph
    unigram = get_top_n_words(data_df['Complaint_clean'], 1,10)
    for word, freq in unigram:
        print(word, freq)
    fig = px.bar(x=[word for word, freq in unigram], y=[freq for word, freq in unigram], title='Top 10 Unigrams')
    fig.write_image("eda_top_10_unigrams.png")

    #Print the top 10 words in the bigram frequency and plot the same using a bar graph
    bigram = get_top_n_words(data_df['Complaint_clean'], 2,10)
    for word, freq in bigram:
        print(word, freq)
    fig = px.bar(x=[word for word, freq in bigram], y=[freq for word, freq in bigram], title='Top 10 Bigrams')
    fig.write_image("eda_top_10_bigrams.png")

    #Print the top 10 words in the trigram frequency and plot the same using a bar graph
    trigram = get_top_n_words(data_df['Complaint_clean'], 3,10)
    for word, freq in trigram:
        print(word, freq)
    fig = px.bar(x=[word for word, freq in trigram], y=[freq for word, freq in trigram], title='Top 10 Trigram')
    fig.write_image("eda_top_10_trigrams.png")

    data_df['Complaint_clean'] = data_df['Complaint_clean'].str.replace('xxxx','')

    return data_df

def feature_extraction(data_df):
    # Use tf idf vectorization - max_df >0.95 and min_df <2
    tf_idf_vec=TfidfVectorizer(max_df=0.98,min_df=2,stop_words='english')
    
    tfidf=tf_idf_vec.fit_transform(data_df['Complaint_clean'])
    
    return tf_idf_vec, tfidf

def topic_modelling(tf_idf_vec, tfidf, data_df):
    #Load your nmf_model with the n_components i.e 5
    num_topics = 5

    #keep the random_state =40
    nmf_model = NMF(n_components=num_topics, random_state=40)
    nmf_model.fit(tfidf)
    print(len(tf_idf_vec.get_feature_names_out()))    
    for index, topic in enumerate(nmf_model.components_):
        print(f'THE TOP 15 WORDS FOR TOPIC #{index} with tf-idf score')
        print([tf_idf_vec.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
        print('\n')
        
    topic_values = nmf_model.transform(tfidf)
    topic_values.argmax(axis=1)
    
    #Assign the best topic to each of the cmplaints in Topic Column
    data_df['Topic'] = topic_values.argmax(axis=1)

    Topic_names = {
        0: 'Bank Account services',
        1: 'Credit card or prepaid card',
        2: 'Others',
        3: 'Theft/Dispute Reporting',
        4: 'Mortgage/Loan'
    }
    #Replace Topics with Topic Names
    data_df['Topic_category'] = data_df['Topic'].map(Topic_names)
    return data_df

def eval_model(training_data, y_test,y_pred,y_pred_proba,labels,type='Training'):
    print(type,'results')
    print('Accuracy: ', accuracy_score(y_test,y_pred))
    print('Precision: ', precision_score(y_test,y_pred,average='weighted'))
    print('Recall: ', recall_score(y_test,y_pred,average='weighted'))
    print('F1 Score: ', f1_score(y_test,y_pred,average='weighted'))
    print('ROC AUC Score: ', roc_auc_score(y_test,y_pred_proba,average='weighted',multi_class='ovr'))
    print('Classification Report: ', classification_report(y_test,y_pred))
    cm = confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {type}")
    plt.savefig(f"confusion_matrix_{type}.png")
    plt.close()

def run_model(model, train_X, train_y, param_grid):
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=40)
    grid=GridSearchCV(model,param_grid={},cv=cv,scoring='f1_weighted',verbose=1,n_jobs=-1)
    grid.fit(train_X,train_y)
    return grid.best_estimator_

def train_and_infer_model(data_df):
    training_data = data_df[['complaint_what_happened','Topic']]
    
    # Vector counts
    count_vect = CountVectorizer()
    #Write your code to get the Vector count
    X_train_counts = count_vect.fit_transform(training_data['complaint_what_happened'])
    #Write your code here to transform the word vector to tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_tf = tfidf_transformer.fit_transform(X_train_counts)
    # Checking for class imbalance
    fig = px.bar(x=training_data['Topic'].value_counts().index, y=training_data['Topic'].value_counts().values/max(training_data['Topic'].value_counts().values), title='Class Imbalance')
    fig.write_image("class_imbalance.png")

    # Prepare the training and test data
    train_X, test_X, train_y, test_y = train_test_split(X_train_tf, training_data['Topic'], test_size=0.2, random_state=40)

    #running and evaluating the Logistic Regression model
    params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 200, 300, 500, 1000],
        'class_weight': [None, 'balanced']
    }
    log_reg_model=run_model(LogisticRegression(), train_X, train_y, params)
    labels = log_reg_model.classes_
    eval_model(training_data, train_y, log_reg_model.predict(train_X), log_reg_model.predict_proba(train_X), labels, type='Training-log_reg')
    eval_model(training_data, test_y, log_reg_model.predict(test_X), log_reg_model.predict_proba(test_X), labels, type='Test-log_reg')


    #running and evaluating the Decision Tree model
    params = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 2, 4, 6, 8, 10],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }
    dc_tree_model=run_model(DecisionTreeClassifier(), train_X, train_y, params)
    labels = dc_tree_model.classes_
    eval_model(training_data, train_y, dc_tree_model.predict(train_X), dc_tree_model.predict_proba(train_X), labels, type='Training-dec-tree')
    eval_model(training_data, test_y, dc_tree_model.predict(test_X), dc_tree_model.predict_proba(test_X), labels, type='Test-dec-tree')

    prediction(count_vect, tfidf_transformer, log_reg_model)
    # return log_reg_model, count_vect, tfidf_transformer
    #running and evaluating the Random Forest model
    # params = {
    #     'n_estimators': [10, 50, 100, 200, 500],
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 2, 4, 6, 8, 10],
    #     'min_samples_split': [2, 4, 6, 8, 10],
    #     'min_samples_leaf': [1, 2, 4, 6, 8, 10],
    #     'max_features': [None, 'auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False]
    # }
    # model=run_model(RandomForestClassifier(), train_X, train_y, params)
    # labels = model.classes_
    # eval_model(training_data, train_y, model.predict(train_X), model.predict_proba(train_X), labels, type='Training-ran-forst')
    # eval_model(training_data, test_y, model.predict(test_X), model.predict_proba(test_X), labels, type='Test-ran-forst')

    # #running and evaluating the XGBoost model
    # params = {
    #     'n_estimators': [100, 200, 500],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'gamma': [0, 0.5, 1],
    #     'min_child_weight': [1, 3, 5],
    #     'subsample': [0.5, 0.8, 1],
    #     'colsample_bytree': [0.5, 0.8, 1]
    # }
    # model=run_model(XGBClassifier(), train_X, train_y, params)
    # labels = model.classes_
    # eval_model(training_data, train_y, model.predict(train_X), model.predict_proba(train_X), labels, type='Training-xgboost')
    # eval_model(training_data, test_y, model.predict(test_X), model.predict_proba(test_X), labels, type='Test-xgboost')

def predict_lr(text, count_vect, tfidf_transformer, model):
    Topic_names = {0:'Account Services', 1:'Others', 2:'Mortgage/Loan', 3:'Credit card or prepaid card', 4:'Theft/Dispute Reporting'}
    X_new_counts = count_vect.transform(text)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = model.predict(X_new_tfidf)
    return Topic_names[predicted[0]]

def prediction(count_vect, tfidf_transformer, input_model):
    # Applying the best model on the Custom Text
    # We will use the XGBoost model as it has the best performance
    df_complaints = pd.DataFrame({'complaints': ["I can not get from chase who services my mortgage, who owns it and who has original loan docs", 
                                    "The bill amount of my credit card was debited twice. Please look into the matter and resolve at the earliest.",
                                    "I want to open a salary account at your downtown branch. Please provide me the procedure.",
                                    "Yesterday, I received a fraudulent email regarding renewal of my services.",
                                    "What is the procedure to know my CIBIL score?",
                                    "I need to know the number of bank branches and their locations in the city of Dubai"]})

    df_complaints['tag'] = df_complaints['complaints'].apply(lambda x: predict_lr([x], count_vect, tfidf_transformer, input_model))
    print(df_complaints)
    df_complaints.to_csv("inference_predicted_complaints.csv", index=False)

def main():
    # Total data is 78k, adjust as needed to speed up the process
    df = load_data(data_num=10000)
    print("Loading data completed")
    df = preprocess_data(df)
    print("Preprocessing data completed")
    
    # Exploratory data analysis
    df = eda(df)
    print("EDA completed")
    
    # feature extraction using TF_IDF
    tf_idf_vec, tfidf = feature_extraction(df)
    
    # Topic modelling using NMF model - clustering
    df = topic_modelling(tf_idf_vec, tfidf, df)
    
    # Supervised model to predict  new complaints to relevant topics
    df = train_and_infer_model(df)


if __name__ == "__main__":
    main()
    print("Analysis completed")

