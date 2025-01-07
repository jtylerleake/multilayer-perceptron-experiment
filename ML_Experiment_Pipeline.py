# -*- coding: utf-8 -*-
'''
contains the ML_Experiment_Pipeline class 

@ name:             ML_Experiment_Pipeline.py
@ last update:      07-29-2024
@ author:           Tyler Leake
'''

import os 
import numpy as np
import matplotlib as plt
from pathlib import Path
from Dataset import Dataset
from CrossValidation import CrossValidation
from MLP import MLP



class ML_Pipeline:
    
    def __init__(self):
        
        self.dataset_directory = r'\UCI-Data'
        self.dataset = None
        
        self.default_ranges = {
        'learning rate'  : np.arange(0, 0.1, 0.10),
        'batch size'     : np.arange(16, 108, 8),
        'epochs'         : np.arange(100, 1500, 100)}
        
    def run_pipeline(self):
        '''command line interface for ML pipeline'''
        
        print('\nLoading Tool...')
        
        self.run_dataset_spec_CL()              # Dataset
        experiment = self.run_experiment_CL()   # Specify, Run Experiment
        self.plot(experiment.scores)            # Plot Results
    
    def run_dataset_spec_CL(self):
        '''
        prompts for a dataset to use; instantiates the dataset; preprocesses 
        the dataset using the config details found in PreprocessingConfig.py
        '''
        
        print('''
              
              \n
              
              The following datasets have been loaded and configured for the 
              tool. Please enter the number of the dataset you wish to use 
              for training and testing:
              
              \n
              
              Warnings:
                  
              \n (1) If you would like to train/test with another dataset, it must 
              first be configured in the PreprocessingConfig.py file 
              
              \n (2) All datasets must be .data files
              
              ''')
              
        # iterate over all files in dataset directory; prompt for the file
        files = os.listdir(self.dataset_directory)
        for i in range(len(files)): print(f'0{str(i)}: {files[i]}')
        dataset_num = input('Dataset No: ')
        dataset_name = files[int(dataset_num)]
        
        # get the remaining path to dataset on the machine
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        filepath = parent_dir + self.dataset_dir + '\\' + dataset_name +'.data'
        filepath = Path(filepath)
        
        # instantiate the Dataset object
        self.dataset = Dataset(dataset_name, filepath)
        print('n\COMPLETE: Dataset Loading')
        
        # preprocess the Dataset
        self.dataset.preprocess_data()
        print('COMPLETE: Dataset Preprocessing')
        
    def run_experiment(self):
        '''
        prompts for an appropriate architecture to make and the 
        hyperparameter ranges to use when tuning the network
        '''
        
        if self.dataset.objective == 'regression': 
            first_architecture = 'Linear Regression Network'
        
        if self.dataset.objective == 'classification': 
            first_architecture = 'Logistic Regression Network'
        
        print(f'''
              
              \n
              
              The {self.dataset.name} dataset is designed for 
              {self.dataset.objective} tasks.
              
              The following archiectures are available to train: \n
                  
              01:  {first_architecture}
              02:  Multilayer Perceptron (MLP)
              
              \n 
              
              Please enter the number corresponding to the architecture
              you wish to train 
              
              ''')
              
        arch_no = int(input('\nArchitecture No: '))
        
        if arch_no == 1:
            
            init = input('Initialization Method: ')
            activation = input('Activation Function: ')
            loss = input('Loss Function: ')
            
            model = MLP(self.dataset, 
                        layer_dimensions = 1,
                        layer_initializations = init,
                        layer_activation_fns = activation,
                        layer_loss_fns = loss)
                         
            experiment = CrossValidation(model,
                                         k_folds = 5,
                                         val_size = 0.20,
                                         hyperparam_ranges = self.default_ranges,
                                         sample_size = 100,
                                         run_experiment=True)
            
            return experiment 
            
        if arch_no == 2:

            print('Enter Values Comma Separated: ')
            
            dims = input('Network Layer Widths: ')
            inits = input('Initialization Methods: ')
            activations = input('Activation Functions: ')
            losses = input('Loss Functions: ')
            
            model = MLP(self.dataset, 
                        layer_dimensions = dims,
                        layer_initializations = inits,
                        layer_activation_fns = activations,
                        layer_loss_fns = losses)
                         
            experiment = CrossValidation(model,
                                         k_folds = 5,
                                         val_size = 0.20,
                                         hyperparam_ranges = self.default_ranges,
                                         sample_size = 100,
                                         run_experiment=True)

            return experiment
            
    def plot(self, scores):
        '''plot the evaluation scores'''
        
        title = 'Model Evaluation Scores'
        labels = [f"Model {i+1}" for i in range(len(scores))]
        
        avg_score = np.mean(scores)
        
        scores_with_avg = scores + [avg_score]
        labels_with_avg = labels + ['Average']
    
        plt.figure(figsize=(10, 6))
        plt.bar(labels_with_avg, scores_with_avg, color='skyblue', edgecolor='black')
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        
if '__name__' == '__main__':
    ML_Pipeline()
    ML_Pipeline.run_pipeline()