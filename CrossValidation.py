# -*- coding: utf-8 -*-
'''
contains the CrossValidation class used as the framework for model experiments.
models are trained and evaluated using kx2 cross validation

@ name:           CrossValidation.py
@ last update:    07-29-2024
@ author:         Tyler Leake
'''

import numpy as np
import Dataset
import MLP
from random import sample
from statistics import mean



class CrossValidation(Dataset):
    
    def __init__(self,
                 model,
                 k_folds, 
                 val_size, 
                 hyperparam_ranges,
                 sample_size, 
                 run_experiment=False):
        
        '''
        implementation of kx2 cross validation experiment
        
        params:
        
        k_folds (int); number of folds for kx2 cross-validation
        val_size (float); % of dataset to use for validation purposes
        hyperparam_ranges (dict); hyperparams (key) and ranges to sample (value)
        sample_size (int); number of sample to draw for random search
        run_experiment (bool); if True, run full cross-validation experiment
        '''
        
        self.model = model
        self.k = k_folds    # number of folds to use for kx2 CV
        self.X = None       # training set
        self.Z = None       # validation set
        
        self.scores = None  # performance metrics
        
        # 
        # Hyperparameter Tuning: 
        # 
        # hyperparam_range: sample ranges for each hyperparameter to sample
        # sample_size: number of samples to draw in random search method
        # best_hyperparam_setting: best setting found by random search method
        # 
        
        self.hyperparam_range = hyperparam_ranges
        self.sample_size = sample_size
        self.best_hyperparam_setting = None
        
        if run_experiment: self.run_experiment(val_size)
    
    def run_experiment(self, val_set_pct):
        '''
        overhead experimental procedure for each model
        
        steps:

        (1) separate X% of the data as the validation set (val_set_pct)
        (2) run kx2 cross validation method for each candidate hyperparam set
        (3) select the best hyperparam set (criteria = avg. performance)
        (4) run kx2 cross validation method on training set with best hyperparams
        (5) report the average performance of the kx2 experiments
        
        params:
        validation_set_pct (float); % of dataset to designate for tuning
        
        returns:
        none; self.scores attribute updated 
        '''
        
        # split the data into a training set and a validation set
        self.val, self.train = self.split_stratified(self.data, val_set_pct)
        
        # tune hyperparams via random search w/ cross validation
        self.best_hyperparam_setting = self.random_search()
        
        # run cross validation using the best setting
        self.scores = self.cross_validate()
        
    def split_stratified(self, dataset, partition1_pct):
        '''
        splits a Dataset into two partitions of random samples. if the data is 
        for classification, the partitions will be stratified by class label
        
        partition sizes:
                
        partition 1 (pt1) = num indices * partition1_pct
        partition 2 (pt2) = num indices * (1 - partition1_pct)

        params: 
        partition1_pct (float); designated size of partition1 expressed as a % 
            
        return:
        pt1, pt2 (tuple); partition1 and partition2 np arrays
        '''
        
        if self.learning_type == 'Classification':
            
            # split dataset into a features series (X) and a class series (Y)
            # retrieve the unique classes from the class series
            
            X = np.delete(dataset, dataset.response_idx, axis=1)
            Y = dataset[:, dataset.response_idx]
            unique_classes = np.unique(Y)
            
            pt1_list, pt2_list = [], [] # initialize lists for partitions 1 & 2
            
            # iterate over unique class labels; for each class, filter the data 
            # and select appropriate number of random indices for each partition
            
            for label in unique_classes:
                
                # filter data by class; shuffle
                class_indices = np.where(Y == label)[0]
                np.random.shuffle(class_indices)
                
                # split the indices based on partitions sizes
                num_pt1 = int(len(class_indices) * partition1_pct)
                pt1_indices = class_indices[:num_pt1]
                pt2_indices = class_indices[num_pt1:]
                
                # append samples to the pt1 and pt2 lists
                pt1_list.append(X[pt1_indices])
                pt2_list.append(Y[pt2_indices])
            
            # concatenate splits for all classes into the two partitions; return
            pt1 = np.concatenate(pt1_list, axis=0)
            pt2 = np.concatenate(pt2_list, axis=0)
            return pt1, pt2
                
        if self.learning_type == 'Regression':
            
            np.random.shuffle(self.data) # shuffle the data
            
            # split the data by indices based on partitions sizes; return
            num_pt1 = int(len(self.data) * partition1_pct)
            pt1 = self.data[:num_pt1]
            pt2 = self.data[num_pt1:]
            return pt1, pt2
    
    def random_search(self):
        '''
        implementation of random search for hyperparameter tuning
        
        return:
        best_setting (dict); best hyperparameter setting from cross-validation
        '''
        
        # draw hyperparameter samples
        param_samples = self.draw_hyperparam_samples() 
        
        # for each hyperparameter sample, run cross validation method and 
        # train models; collect the average performance metrics for each
        
        scores = []
        
        for hyperparams in param_samples:
            score = self.cross_validate(self.X, hyperparams, True)
            scores.append(score)
            
        best_score_idx = scores.index(max(scores))
        best_setting = param_samples[best_score_idx]
        return best_setting

    def draw_hyperparam_samples(self):
        '''
        generates random samples of hyperparameters using the hyperparam_ranges 
        attribute. used by the random search method to tune models
        
        return:
        parameter_samples (list); list of hyperparam settings (dicts) to sample
        '''
        
        parameter_samples = []
        
        for _ in range(self.sample_size):
            
            sampled_params = {}
        
            # draw a random sample for each hyperparameter
            for param, param_range in self.hyperparam_ranges.items():
                param_sample = np.random.choice(param_range)
                sampled_params[param] = param_sample

            parameter_samples.append(sampled_params)

        return parameter_samples
    
    def cross_validate(self, dataset, hyperparams, tuning=False):
        '''
        implementation of kx2 cross validation 
        
        params:
        dataset (np array); data to train with (and test with if not tuning)
        hyperparams (dict); hyperparameter settings
        tuning (bool); True IF doing hyperparameter tuning (val set used to test)
        
        return:
        scores (list): list of performance scores for each model trained
        mean (int): mean performance score across all models
        '''
        
        #
        # Procedure
        #
        # partition the dataset into halves and train a model on each fold; if
        # tuning hyperparams, test the model on the held out validation set
        # otherwise test on other fold; record score; reset model; continue
        #
        
        scores = []
        
        for cycle in range(self.k):
            
            fold1, fold2 = self.split_stratified(dataset, 0.5)
        
            for i, fold in enumerate([fold1, fold2]):
                self.model.train(fold, hyperparams)
                score = self.model.eval(self.Z if tuning else fold2 if i == 0 else fold1)
                scores.append(score) 
                self.model.reset()
                
        avg_scores = mean(scores)
        return scores, avg_scores
                
            