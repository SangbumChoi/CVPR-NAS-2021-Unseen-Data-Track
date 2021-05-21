from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
]

PRIMITIVES_reduce = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'bin_dil_conv_3x3',
    'bin_dil_conv_5x5',
    'skip_connect',
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

latest_cell_zeroise = Genotype(
    normal=[('none', 1), ('bin_dil_conv_3x3', 0), ('bin_conv_3x3', 1), ('none', 2), ('none', 2), ('none', 3),
            ('none', 3), ('none', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('none', 1), ('max_pool_3x3', 0), ('none', 2), ('none', 3), ('none', 2), ('none', 3),
            ('none', 4)], reduce_concat=range(2, 6))

latest_cell = Genotype(
    normal=[('bin_conv_3x3', 1), ('bin_dil_conv_3x3', 0), ('bin_conv_3x3', 1), ('bin_conv_3x3', 2), ('bin_conv_3x3', 2),
            ('bin_dil_conv_3x3', 1), ('bin_dil_conv_3x3', 1), ('bin_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('bin_conv_3x3', 2), ('max_pool_3x3', 0),
            ('bin_dil_conv_3x3', 3), ('max_pool_3x3', 0), ('bin_dil_conv_3x3', 4)], reduce_concat=range(2, 6))

latest_cell_skip1 = Genotype(
    normal=[('none', 1), ('bin_dil_conv_3x3', 0), ('bin_conv_3x3', 1), ('skip_connect', 0), ('none', 2), ('none', 2), ('none', 3),
            ('none', 3), ('none', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('none', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('none', 2), ('none', 3), ('none', 2), ('none', 3),
            ('none', 4)], reduce_concat=range(2, 6))

latest_cell_skip2 = Genotype(
    normal=[('none', 1), ('bin_dil_conv_3x3', 0), ('bin_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('none', 2), ('none', 2), ('none', 3),
            ('none', 3), ('none', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('none', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('none', 2), ('none', 3), ('none', 2), ('none', 3),
            ('none', 4)], reduce_concat=range(2, 6))

latest_cell_skip3 = Genotype(
    normal=[('sep_conv_3x3', 1), ('bin_dil_conv_3x3', 0), ('bin_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('none', 2), ('none', 2), ('none', 3),
            ('none', 3), ('none', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('none', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('bin_conv_3x3', 2), ('bin_dil_conv_3x3', 3), ('none', 2), ('none', 3),
            ('none', 4)], reduce_concat=range(2, 6))

latest_cell_skip4 = Genotype(
    normal=[('sep_conv_3x3', 1), ('bin_dil_conv_3x3', 0), ('bin_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('none', 2), ('none', 2), ('none', 3),
            ('none', 3), ('none', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('none', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('bin_conv_3x3', 2), ('bin_dil_conv_3x3', 3), ('none', 2), ('none', 3),
            ('none', 4)], reduce_concat=range(2, 6))