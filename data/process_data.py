import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets then merge them together.
    
    Arguments:
    messages_filepath: string. Messages filepath.
    categories_filepath: string. Categories filepath.
       
    Outputs:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    Split categories into separate category columns, 
    convert category values to just numbers 0 or 1 and drop duplicates.
    
    Arguments:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
       
    Outputs:
    df: dataframe. Dataframe containing cleaned version of given dataframe.
    """
  
    # split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[[1]] 
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]    

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)    

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save cleaned data into an SQLite database.
    
    Arguments:
    df: dataframe. Dataframe containing cleaned merged messages and 
    categories datasets.
    database_filename: string. Filename for output database.
       
    Outputs:
    None
    """  
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('responses', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()