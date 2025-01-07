# -*- coding: utf-8 -*-
'''
configuration dictionary for datasets. all data used by program to be
configured here (currently only datasets in 'UCI-Data' are configured).

@ name:           PreprocessingConfig.py
@ last update:    07-29-2024
@ author:         Tyler Leake
'''



'''

Dataset Links
-----------------

Abalone:
archive.ics.uci.edu/dataset/1/abalone

Breast Cancer Wisconsin (Diagnostic):
archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Car Evaluation:
archive.ics.uci.edu/dataset/19/car+evaluation

Forest Fires:
archive.ics.uci.edu/dataset/162/forest+fires

Congressional Voting Records:
archive.ics.uci.edu/dataset/105/congressional+voting+records

Computer Hardware:
archive.ics.uci.edu/dataset/29/computer+hardware

'''



Config = {
    

    'Abalone' : {
        
        'Objective'        : 'regression', 
        'Response Var'     : 'rings',
        'Extranneous Cols' : ['sex'],
    
        'Feature Map' : {
        
        'Sex'              : [None, None],
        'Length'           : ['numeric', None],
        'Diameter'         : ['numeric', None],
        'Height'           : ['numeric', None],
        'Whole_weight'     : ['numeric', None],
        'Shucked_weight'   : ['numeric', None],
        'Viscera_weight'   : ['numeric', None],
        'Shell_weight'     : ['numeric', None],
        'Rings'            : ['numeric', None]
        
        }, 

        'Impute Information' : {
        
        'Sex'              : ['mean', None],
        'Length'           : ['mean', None],
        'Diameter'         : ['mean', None],
        'Height'           : ['mean', None],
        'Whole_weight'     : ['mean', None],
        'Shucked_weight'   : ['mean', None],
        'Viscera_weight'   : ['mean', None],
        'Shell_weight'     : ['mean', None],
        'Rings'            : ['mean', None],

        },
    },
    

    'Breast Cancer Wisconsin (Diagnostic)' : {
        
        'Objective'        : 'classification', 
        'Response Var'     : 'class',
        'Extranneous Cols' : ['sample_code_number'],
    
        'Feature Map': {
            
        'Sample_code_number'           : ['numeric', None], 
        'Clump_thickness'              : ['numeric', None], 
        'Uniformity_of_cell_size'      : ['numeric', None], 
        'Uniformity_of_cell_shape'     : ['numeric', None], 
        'Marginal_adhesion'            : ['numeric', None], 
        'Single_epithelial_cell_size'  : ['numeric', None], 
        'Bare_nuclei'                  : ['numeric', None], 
        'Bland_chromatin'              : ['numeric', None], 
        'Normal_nucleoli'              : ['numeric', None], 
        'Mitoses'                      : ['numeric', None], 
        'Class'                        : ['nominal', None]
        
        }, 
        
        'Impute Information' : {
            
        'Sample_code_number'           : ['mean', None], 
        'Clump_thickness'              : ['mean', None], 
        'Uniformity_of_cell_size'      : ['mean', None], 
        'Uniformity_of_cell_shape'     : ['mean', None], 
        'Marginal_adhesion'            : ['mean', None], 
        'Single_epithelial_cell_size'  : ['mean', None], 
        'Bare_nuclei'                  : ['mean', ['?']], 
        'Bland_chromatin'              : ['mean', None], 
        'Normal_nucleoli'              : ['mean', None], 
        'Mitoses'                      : ['mean', None], 
        'Class'                        : ['mean', None], 
        
        }
    },
    

    'Car Evaluation' : {
        
        'Objective'         : 'classification', 
        'Response Var'      : 'class',
        'Extranneous Cols'  :  [],
        
        'Feature Map': {
            
        'Buying'   : ['nominal', ['vhigh', 'high', 'med', 'low']],
        'Maint'    : ['nominal', ['vhigh', 'high', 'med', 'low']],
        'Doors'    : ['nominal', ['2', '3', '4', '5more']], 
        'Persons'  : ['nominal', ['2', '3', 'more']], 
        'Lug_boot' : ['nominal', ['small', 'med', 'big']], 
        'Safety'   : ['nominal', ['low', 'med', 'high']],
        'Class'    : ['nominal', ['unacc', 'acc', 'good', 'vgood']],
        
        },
        
        'Impute Information': {
            
        'Buying'   : ['mode', None],
        'Maint'    : ['mode', None],
        'Doors'    : ['mode', None],
        'Persons'  : ['mode', None],
        'Lug_boot' : ['mode', None],
        'Safety'   : ['mode', None],
        'Class'    : ['mode', None],
        
        }
    },
    
        
    'Forest Fires' : {
        
        'Objective'         : 'regression', 
        'Response Var'      : 'area',
        'Extranneous Cols'  : [],
    
        'Feature Map': {
                        
        'X'       : ['numeric', None],
        'Y'       : ['numeric', None],
        'month'   : ['nominal', None],
        'day'     : ['nominal', None],
        'FFMC'    : ['numeric', None],
        'DMC'     : ['numeric', None],
        'DC'      : ['numeric', None],
        'ISI'     : ['numeric', None],
        'temp'    : ['numeric', None],
        'RH'      : ['numeric', None],
        'wind'    : ['numeric', None],
        'rain'    : ['numeric', None],
        'area'    : ['numeric', None],
        
        },
            
        'Impute Information': {
            
        'X'       : ['mean', None],
        'Y'       : ['mean', None],
        'month'   : ['mode', None],
        'day'     : ['mode', None],
        'FFMC'    : ['mean', None],
        'DMC'     : ['mean', None],
        'DC'      : ['mean', None],
        'ISI'     : ['mean', None],
        'temp'    : ['mean', None],
        'RH'      : ['mean', None],
        'wind'    : ['mean', None],
        'rain'    : ['mean', None],
        'area'    : ['mean', None],

        }
    },
    
    
    'Congressional Voting Records' : {
        
        'Objective'        : 'classification', 
        'Response Var'     : 'class',
        'Extranneous Cols' : [],
    
        'Feature Map': {
            
        'Class'                                   : ['nominal', None],
        'Handicapped_infants'                     : ['nominal', None], 
        'Water_project_cost_sharing'              : ['nominal', None], 
        'Adoption_of_the_budget_resolution'       : ['nominal', None], 
        'Physician_fee_freeze'                    : ['nominal', None], 
        'El_salvador_aid'                         : ['nominal', None], 
        'Religious_groups_in_schools'             : ['nominal', None], 
        'Anti_satellite_test_ban'                 : ['nominal', None], 
        'Aid_to_nicaraguan_contras'               : ['nominal', None], 
        'Mx_missile'                              : ['nominal', None], 
        'Immigration'                             : ['nominal', None], 
        'Synfuels_corporation_cutback'            : ['nominal', None], 
        'Education_spending'                      : ['nominal', None], 
        'Superfund_right_to_sue'                  : ['nominal', None], 
        'Crime'                                   : ['nominal', None], 
        'Duty_free_exports'                       : ['nominal', None], 
        'Export_administration_act_south_africa'  : ['nominal', None], 
        
        }, 
        
        'Impute Information': {
            
        'Class'                                   : ['mode', None], 
        'Handicapped_infants'                     : ['mode', None], 
        'Water_project_cost_sharing'              : ['mode', None], 
        'Adoption_of_the_budget_resolution'       : ['mode', None], 
        'Physician_fee_freeze'                    : ['mode', None], 
        'El_salvador_aid'                         : ['mode', None], 
        'Religious_groups_in_schools'             : ['mode', None], 
        'Anti_satellite_test_ban'                 : ['mode', None], 
        'Aid_to_nicaraguan_contras'               : ['mode', None], 
        'Mx_missile'                              : ['mode', None], 
        'Immigration'                             : ['mode', None], 
        'Synfuels_corporation_cutback'            : ['mode', None], 
        'Education_spending'                      : ['mode', None], 
        'Superfund_right_to_sue'                  : ['mode', None], 
        'Crime'                                   : ['mode', None], 
        'Duty_free_exports'                       : ['mode', None], 
        'Export_administration_act_south_africa'  : ['mode', None], 

        }
    },
    
    
    'Computer Hardware' : {
        
        'Objective'         : 'regression', 
        'Response Var'      : 'ERP',
        'Extranneous Cols'  : ['VendorName', 'ModelName'],    
    
        'Feature Map': {
            
        'VendorName'  : [None, None],
        'ModelName'   : [None, None],
        'MYCT'        : ['numeric', None],
        'MMIN'        : ['numeric', None],
        'MMAX'        : ['numeric', None],
        'CACH'        : ['numeric', None],
        'CHMIN'       : ['numeric', None],
        'CHMAX'       : ['numeric', None],
        'PRP'         : ['numeric', None],
        'ERP'         : ['numeric', None],
        
        }, 
        
        'Impute Information': {
            
        'VendorName'  : ['mean', None],
        'ModelName'   : ['mean', None],
        'MYCT'        : ['mean', None],
        'MMIN'        : ['mean', None],
        'MMAX'        : ['mean', None],
        'CACH'        : ['mean', None],
        'CHMIN'       : ['mean', None],
        'CHMAX'       : ['mean', None],
        'PRP'         : ['mean', None],
        'ERP'         : ['mean', None],
        
        }
    },
}
