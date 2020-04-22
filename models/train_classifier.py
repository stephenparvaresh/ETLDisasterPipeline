import sys
import pandas as pd
import numpy as np
import os
import pickle

from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import hmean
from scipy.stats.mstats import gmean

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Input:
        database_filepath: File path of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for 36 categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names
                         

def tokenize(text):
    '''
    Tokenize and clean text
    Input:
        text: original message text
    Output:
        lemmed: Tokenized, cleaned, and lemmatized text
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Extract starting verb from text.
    Input:
        text: original message text
    Output:
        tagged training data
    '''
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        Results of GridSearchCV
    '''
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
#         'clf__estimator__n_estimators': [50, 100, 200],
#         'clf__estimator__min_samples_split': [2, 3, 4],
#         'clf__estimator__criterion': ['entropy', 'gini']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    def multioutput_fscore(Y_true,Y_pred,beta=1):
        '''
        Evaluate multioutput model performance
        Input:
            y_true: instance of y in dataframe
            y_pred: instance of predicted
        Output:
            overall_accuracy: average accuracy of all features
            multi_f1: f1 score of multioutput
        '''
        
        score_list = []
        if isinstance(Y_pred, pd.DataFrame) == True:
            Y_pred = Y_pred.values
        if isinstance(y_true, pd.DataFrame) == True:
            Y_true = Y_true.values
        for column in range(0,Y_true.shape[1]):
            score = fbeta_score(Y_true[:,column],Y_pred[:,column],beta,average='weighted')
            score_list.append(score)
        f1score_numpy = np.asarray(score_list)
        f1score_numpy = f1score_numpy[f1score_numpy<1]
        f1score = gmean(f1score_numpy)
        return  f1score
    Y_pred = model.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))
    
    Y_pred_pd = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for column in y_test.columns:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column],Y_pred_pd[column]))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    filename = 'classifier.sav'
    pickle.dump(model, open(filename, 'wb'))

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