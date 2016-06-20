# In this file, I take a data frame and clean up the data to make it
# more manageable, eventually returning the dataframe
import numpy as np
import pandas as pd

#Creates an ID column for the dataframe
def create_id_feature(dframe):
    dframe['ID'] = dframe['AnimalID'].map(lambda x: int(x[1:]))
    return dframe

# Create a feature to determine if a breed is mixed
def create_mixed_feature(df):
    ismix = lambda x: x in str(x).lower()
    df['Mixed'] =  df['Breed'].map(ismix).astype(np.int)
    return df

# Binary integer feature if an animal is cat or dog (0 == dog, 1 == cat)
def create_cat_feature(df):
    df['Cat'] = (df.AnimalType == 'Cat').astype(np.int)
    return df

# Creates feature for if a cat is a domestic hair mix
def create_hairmix_feature(df):
    valids = ['Domestic Shorthair Mix',\
              'Domestic Medium Hair Mix',\
              'Domestic Longhair Mix',\
              'Domestic Shorthair',\
              'Domestic Longhair',\
              'Domestic Medium Hair']

    df['Hairmix'] = (df.Breed.isin(valids).astype(np.int)
    return df

fixed = {'Neutered Male': 1, 'Spayed Female': 1, 'Intact Male': 0,\
         'Intact Female': 0, 'Unknown': np.nan, np.nan: np.nan}

# Create feature to tell if animal has been spayed/neutered
def create_neutered_feature(df):
    df['Neutered'] = df.SexuponOutcome.map(fixed)
    return df

def create_age_feature(df):
    def age_in_years(age):
        if type(age) == float:
            return np.nan
        ages = age.split()
        num = int(ages[0])
        modifier = ages[1]
        d = { 'days': 365, 'day': 365,\
             'month': 12, 'months': 12,\
             'year': 1, 'years': 1,\
             'weeks': 52, 'week': 52 }
        return num / d[modifier]
    df['Age'] = df.AgeuponOutcome.map(age_in_years)
    return df


def modify_DateTime_feature(df):
    df['DateTime'] = df.DateTime.astype(pd.datetime)
    return df

def create_outcome_features(df):
    df = df.join(pd.get_dummies(df.OutcomeType))
    return df

def clean_data(df):
    df = create_id_feature(df)
    df = create_mixed_feature(df)
    df = create_cat_feature(df)
    df = create_neutered_feature(df)
    df = create_age_feature(df)
    df = modify_DateTime_feature(df)
    df = create_outcome_features(df)
    df = create_hairmix_feature(df)
    return df


