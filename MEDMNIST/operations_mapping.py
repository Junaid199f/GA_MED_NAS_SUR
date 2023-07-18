#This file contains the operations used in the search space


operations_mapping = {
    1:'max_pool_3x3',
    2:'avg_pool_3x3',
    3:'skip_connect',
    4:'sep_conv_3x3',
    5:'sep_conv_5x5',
    6:'dil_conv_3x3',
    7:'dil_conv_5x5',
    8:'conv_7x1_1x7',
    9:'inv_res_3x3',
    10:'inv_res_5x5'}

primitives = [
    'max_pool_3x3', 'avg_pool_3x3',
    'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3',
    'dil_conv_5x5', 'conv_7x1_1x7',
    'inv_res_3x3', 'inv_res_5x5',
    ]