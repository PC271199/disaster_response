import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Read csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Join 2 df
    df = messages.merge(categories, how='inner', on='id')

    return df


def clean_data(df):
    # Get keys-values from each rows
    dics = []
    for i in range(len(df)):
        row = df['categories'].iloc[i]
        dic = {}
        for lb in row.split(';'):
            k, v = lb.split('-')[0], lb.split('-')[1]
            dic[k] = int(v)
        dics.append(dic)
    
    # Create df for clean categories
    clean_cate = pd.DataFrame(dics)

    # Concat
    final_df = pd.concat([df, clean_cate], axis=1, join='inner')
    
    # Remove old category col
    del(final_df['categories'])

    # Drop duplicates
    final_df.drop_duplicates(inplace=True)

    return final_df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('clean_data', engine, index=False)  


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