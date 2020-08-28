import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle


def load_data(database_filepath):
    """
    Load datasets from local SQLite database
    
    Arguments:
    database_filepath: string. Filepath for SQLite database containing cleaned messages dataset.
    
    Outputs:
    X: dataframe. Dataframe containing features dataset.
    y: dataframe. Dataframe containing labels dataset.
    col_names: List of strings. List containing category names.
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('responses', con=engine)

    # drop child_alone column because it has only zeros
    df = df.drop(['child_alone'],axis=1)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    col_names = y.columns.values

    return X, y, col_names


def tokenize(text):
    """
    tokenize function: Normalize, tokenize and lemmatize a given text string
    
    Arguments:
        text: string. String containing message for processing
    Outputs:
        clean_tokens: list of strings. List containing normalized and lemmatized word tokens
    """
    
    # use regular expression to detect all urls in the provided text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # replace url with a urlplaceholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'urlplaceholder')

    # extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords
    tokens_stopwords_removed = [w for w in tokens if w not in stopwords.words("english")]
    
    # lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # list of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens_stopwords_removed]
   
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline
    
    Arguments:
    None
    
    Outputs:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the
    model object and finds the best model parameters.
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'tfidf__use_idf' : [True, False],        
              'vect__max_df' : [1.0, 0.8],              
              'clf__estimator__min_samples_split' : [2, 4],
              'clf__estimator__n_estimators' : [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance based on accuracy, precision, recall, F1 score
    
    Arguments:
    model: model object. a given model
    X_test: array. Array containing actual labels.
    Y_test: array. Array containing predicted labels.
    category_names: list of strings. List containing names for each of the predicted fields.
    
    Outputs:
    None
    """
    
    Y_pred = model.predict(X_test)    
  
    for i, column in enumerate(category_names):
        print('______________________________________________________\n')
        print('Column: {}\n'.format(column))
        print(classification_report(Y_test[column], Y_pred[:, i]))
        

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
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