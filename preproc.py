import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessor():
    def __init__(self, train, top_n=50):
        # category binning
        self.top_funders = list(train.groupby('funder').size().sort_values(ascending=False)[:top_n].index)
        self.top_installers = list(train.groupby('installer').size().sort_values(ascending=False)[:top_n].index)
        self.top_wpt_name = list(train.groupby('wpt_name').size().sort_values(ascending=False)[:top_n].index)
        self.top_subvillage = list(train.groupby('subvillage').size().sort_values(ascending=False)[:top_n].index)
        self.top_scheme_name = list(train.groupby('scheme_name').size().sort_values(ascending=False)[:top_n].index)
        self.top_ward = list(train.groupby('ward').size().sort_values(ascending=False)[:top_n].index)
        self.top_scheme_management = list(train.groupby('scheme_management').size().sort_values(ascending=False)[:top_n].index)
        self.top_region_code = list(train.groupby('region_code').size().sort_values(ascending=False)[:top_n].index)

        # scaling
        train['date_recorded'] = pd.to_datetime(train['date_recorded']).astype(int)
        self.date_recorded_scaler = StandardScaler().fit(train[['date_recorded']])
        self.construction_year_scaler = StandardScaler().fit(train[['construction_year']])
        self.population_scaler = StandardScaler().fit(train[['population']])
        self.longitude_scaler = StandardScaler().fit(train[['longitude']])
        self.latitude_scaler = StandardScaler().fit(train[['latitude']])
        self.gps_height_scaler = StandardScaler().fit(train[['gps_height']])
        self.amount_tsh_scaler = StandardScaler().fit(train[['amount_tsh']])

    def preprocess(self, df, train=False):
        # remove redundant information
        df = df[df.columns.drop(list(df.filter(regex='_type$')))]
        df = df[df.columns.drop(list(df.filter(regex='_group$')))]
        df.drop('recorded_by', axis=1, inplace=True)

        # scaling
        df['date_recorded'] = pd.to_datetime(df['date_recorded']).astype(int)
        df['date_recorded'] = self.date_recorded_scaler.transform(df[['date_recorded']])
        df['construction_year'] = self.construction_year_scaler.transform(df[['construction_year']])
        df['population'] = self.population_scaler.transform(df[['population']])
        df['longitude'] = self.longitude_scaler.transform(df[['longitude']])
        df['latitude'] = self.latitude_scaler.transform(df[['latitude']])
        df['gps_height'] = self.gps_height_scaler.transform(df[['gps_height']])
        df['amount_tsh'] = self.amount_tsh_scaler.transform(df[['amount_tsh']])

        # prepare for dummification
        df['region_code'] = df['region_code'].astype(str)
        df['district_code'] = df['district_code'].astype(str)

        # categorical binning
        df['funder'] = df['funder'].apply(lambda x: x if x in self.top_funders else 'other')
        df['installer'] = df['installer'].apply(lambda x: x if x in self.top_installers else 'other')
        df['wpt_name'] = df['wpt_name'].apply(lambda x: x if x in self.top_wpt_name else 'other')
        df['subvillage'] = df['subvillage'].apply(lambda x: x if x in self.top_subvillage else 'other')  
        df['scheme_name'] = df['scheme_name'].apply(lambda x: x if x in self.top_scheme_name else 'other')
        df['ward'] = df['ward'].apply(lambda x: x if x in self.top_ward else 'other')
        df['region_code'] = df['region_code'].apply(lambda x: x if x in self.top_region_code else 'other')
        df['scheme_management'] = df['scheme_management'].apply(lambda x: x if x in self.top_scheme_management else 'other')
        # dummify
        df = pd.get_dummies(df)
        
        # save cols
        if train == True:
            self.colset = df.columns
        # add in null features
        else:
            for col in self.colset:
                if not col in df.columns:
                    df[col] = 0        
        
        return df