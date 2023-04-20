operations_mapping = { 
    0:'max_pool_3x3',
    1:'avg_pool_3x3',
    2:'skip_connect',
    3:'sep_conv_3x3',
    4:'sep_conv_5x5',
    5:'dil_conv_3x3',
    6:'dil_conv_5x5',
    7:'conv_7x1_1x7',
    8:'inv_res_3x3',
    9:'inv_res_5x5'}

primitives = [
    'max_pool_3x3', 'avg_pool_3x3',
    'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3',
    'dil_conv_5x5', 'conv_7x1_1x7',
    'inv_res_3x3', 'inv_res_5x5',
    ]