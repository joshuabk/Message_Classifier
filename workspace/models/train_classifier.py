import sys


from sklearn.utils.multiclass import type_of_target
import pandas as pd 
from sqlalchemy import create_engine
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle

# load data from data base

def load_data(database_filepath):

    '''Input the file path for the database
       return the X, Y and columns list'''
    engine = create_engine('sqlite:///'+database_filepath)

    df = pd.read_sql('SELECT * FROM messages', engine)

    df.columns
    X = df['message']
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    print(Y.columns.tolist())
    return X,Y, Y.columns.tolist()

# tokenize the text
def tokenize(text):
    '''Input is the text to be tokenized
       returns the tokenized text'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    #create pipline

    # returns model that is grid search opitmized
    # the optimization took an unreasonable amount of time
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    #use grid search
    parameters = {
       #'moc__estimator__max_features':[40, 20],#,20],
        'moc__estimator__n_estimators': [5,10],#, 200],
        'moc__estimator__min_samples_split': [3,5]#, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

   
    return cv
    

# look at metrics of the model
def evaluate_model(model, X_test, Y_test, category_names):
    '''Input is the model and testing data
       prints the evaluation meterics'''
       
    y_pred = model.predict(X_test)
    print('Print predictions')
    print(y_pred)
    print(len(category_names))
    print(Y_test)
    print( y_pred[0])
    Y_test = Y_test.values
    for i in range(len(category_names)):
        res = classification_report(Y_test[i], y_pred[i])
        print(res)
    
#save model to pickle file
def save_model(model, model_filepath):
    '''saves the model to a pickle file'''


    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
 

#accuracy_score(y_test, y_pred)


def main():

    #runs the functions
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()