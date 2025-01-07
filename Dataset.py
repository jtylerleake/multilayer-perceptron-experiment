# -*- coding: utf-8 -*-
'''
contains the Dataset class with methods to prepare data for training/testing
ml algorithms, including data imputation and handling of categorical values

@ name:           Dataset.py
@ last update:    07-29-2024
@ author:         Tyler Leake
'''

import pandas as pd
import numpy as np
from statistics import mean, mode
from DatasetConfig import Config



class Dataset: 
    
    def __init__(self, name, path, preprocess=False):
        
        self.name = name
        self.data = pd.read_csv(path)
        
        # configuration attributes
        config = Config[name]
        self.objective = config['Objective']
        self.extra_cols = config['Extranneous Cols']
        self.response_var = config['Response Var']
        self.feature_map = config['Feature Map']
        self.impute_info = config['Impute Techniques']
        
        # other 
        self.num_features = None
        self.num_classes = None
        self.response_idx = None
        self.stored_headers = None
        
        # preprocess dataset if specified
        if preprocess: self.preprocess_data()
        
    def preprocess_data(self):
        '''
        prepares dataset for training/testing by handling categorical data, 
        replacing missing values, normalizing data and converting to np array
        
        params: 
        none; uses self.data as dataframe object
        
        return: 
        none; update self.data directly
        '''
        
        for col in self.data.columns:
            
            # retrieve feature map and data impute information
            feature_type = self.feature_map[col][0]
            ordinal_vals = self.feature_map[col][1]
            impute_type  = self.impute_info[col][0]
            missing_vals = self.impute_info[col][1]
            
            contains_missing_vals = False if missing_vals == None else True
            
            # remove extranneous cols
            if col in self.extra_cols: 
                self.data = self.data.drop([col], axis=1)

            # Categorical Data Handling:
            # replace missing values with mean or mode, then hot encode 
            # (nominal features) or label encode (ordinal features)
            
            if feature_type in ['ordinal', 'nominal']:
                
                if contains_missing_vals:
                    self.impute_data(col, impute_type, missing_vals)
                
                if feature_type == 'ordinal':
                    self.label_encode(col, ordinal_vals)
                    
                if feature_type == 'nominal':
                    is_target = (col == self.response_var)
                    self.hot_encode(col, is_target)
                
            # Nominal Data Handling:
            # replace missing values with mean or mode, then perform
            # z-score normalization on the data
            
            if feature_type == 'numeric':
                
                if contains_missing_vals:
                    self.impute_data(col, impute_type, missing_vals)
                
                if col != self.response_var:
                    self.z_score_normalize(col)
                
        # set num_features and num_labels attributes. then convert to 
        # np array and store headers in stored_headers attribute
        self.set_num_classes()
        self.set_num_dimensions()
        self.stored_headers = self.data.columns
        self.response_idx = self.stored_headers.get_loc(self.response_var)
        self.data = self.data.to_numpy()
        
    def impute_data(self, col, impute_type, missing_vals):
        '''
        replaces missing values in a column with its mean/mode
        
        params:
        column_num (int): column number to filter and replace data in
        missing_vals (list): missing values map for dataset

        return: 
        none; self.dataset column updated inplace
        '''
        
        # filter column and compute its mean or mode
        col_filter = self.data[~self.data[col].isin(missing_vals)][col]
        col_filter = pd.to_numeric(col_filter)
        
        # replace each missing value mean or mode
        replacement = None
        if impute_type == 'mean': replacement = col_filter.mean()
        if impute_type == 'mode': replacement = col_filter.mode()
        self.data[col].replace(missing_vals, replacement, inplace = True)
    
    def hot_encode(self, col, is_target): 
        '''
        performs one-hot encoding on a single column in the dataset
        
        params:
        col (str): column header to convert to one-hot encoding
            
        return: 
        none; self.dataset updated inplace
        '''
        
        # compute the number of unique values in the column
        unique_vals = self.data.loc[:, col].unique()
        num_unique_vals = len(unique_vals)
        
        # Target Column Encoding
        # create a unique binary vector for each response value in column and
        # map each vector to its corresponding response label
        
        if is_target:
            
            encodings = {}
            
            for i in range(num_unique_vals):
                vector = np.zeros(num_unique_vals)
                vector[i] = 1
                label = unique_vals[i]
                encodings[label] = vector
                
            self.data[col] = self.data[col].map(encodings); return
            
        # Non-Target Column Encoding
        # add a column to the df for each unique value in the original column. 
        # assign a value of 1 to each instance with that value, else 0. 
        
        if not is_target:

            for val in unique_vals:
                col_name = col + '-' + val
                self.data[col_name] = 0
                self.data.loc[self.data[col]==val, col_name] = 1

            self.data.drop(columns=[col], inplace=True); return
            
    def label_encode(self, col, ordinal_vals):
        '''
        performs label encoding on a single column in the dataset
        
        params:
        col (str): header of column to process
        ordinal_values (list): list of ordinal values in col to replace

        return: 
        none; self.dataset column updated inplace
        '''
        
        # replace each categorical value with the the numeric label; 
        # (assumes 'ordinal_vales' list is provided in ascending order)
        for i in range(len(ordinal_vals)):
            self.data[col].replace([ordinal_vals[i]], str(i), inplace=True)
        
    def z_score_normalize(self, col):
        '''
        normalizes a column in the dataset using z-score normalization
        
        params: 
        col (str): header of column to normalize
        
        return: 
        none; self.data updated directly
        '''
        
        # compute the mean and standard deviation of the col 
        # and create a vector of same length as col for each
        col_vector = pd.to_numeric(self.data[col])
        mean = col_vector.mean()
        stdev = np.std(col_vector)
        mean_vector = np.full(len(col_vector), mean)
        stdev_vector = np.full(len(col_vector), stdev)
        
        # compute the normalized vector and update the df
        normalized_vector = (col_vector - mean_vector) / stdev_vector
        self.data[col] = normalized_vector

    def set_n_classes(self):
        '''sets n_classes attr. with int class count after preprocessing'''
        if self.objective == 'classification':
            unique_labels = {tuple(vector) for vector in self.data[self.response_var].values}
            self.num_classes = len(unique_labels)
    
    def set_n_features(self):
        '''sets n_features attr. with int feature count after preprocessing'''
        self.n_features = self.data.shape[1] - 1 # subtract 1 for target col
    
    def is_hot_encoded(self, col):
        '''
        returns true/false if a column has been hot encoded
        
        params:
        col (int); column number in np array or dataframe
        
        return:
        (bool); True if column has been hot-encoded
        '''
        
        col_str = self.stored_headers[col]
        col_dtype = Config[self.name]['Feature Map'][col_str][0]
        
        if col_dtype == 'nominal': return True
        else: return False
        