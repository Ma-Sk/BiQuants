from environs import Env
from marshmallow.validate import OneOf

env: Env = Env()

TASK_ID = env.int('TASK_ID', None)
DEBUG = env.bool('DEBUG', True)
CPU_COUNT = env.int('CPU_COUNT', 6)
LOADER = env.str('LOADER', 'local')
ALGORITHM_ID = env.int('ALGORITHM_ID', 1)

# PATHS
ROOT_PATH = env.str('ROOT_PATH', '')
INPUT_DATA_PATH = env.str('INPUT_DATA_PATH', 'segmentData/')
TECH_PATH = env.str('TECH_PATH', '')
SEGMENT_PATH = env.str('SEGMENT_PATH', 'segmentData/')
OUTPUT_PATH = env.str('OUTPUT_PATH', 'output/')

# AWS CREDS
AWS_ACCESS_KEY_ID = env.str('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = env.str('AWS_SECRET_ACCESS_KEY', '')

# RESULT SETTINGS
SEGMENT_SIZE = env.int('SEGMENT_SIZE', 0)
RESULT_FORMAT = env.str(
    'RESULT_FORMAT', 'csv',
    validate=OneOf(
        ['csv', 'parquet'],
        error="RESULT_FORMAT must be one of: {choices}"
    )
)
