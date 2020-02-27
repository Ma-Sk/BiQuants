import os
from pyspark import SparkContext
from pyspark.sql import SparkSession

import settings as settings
from logic.loader import LoaderStrategy
from logic.utils import timeit
from piplenes import prepare_segment  # prepare_data

# Если hadoop_home уже настроен это трока не нужна
os.environ['HADOOP_HOME'] = r"D:\Programms\Spark\spark-2.4.4-bin-hadoop2.7"

def get_spark_memory(cpus: int):
    mapping = {
        4: 11,
        16: 56,
        64: 244
    }
    return f'{mapping.get(cpus, 3)}g'


def get_spark_session(loader_type: str, cpus: int) -> SparkSession:
    """Create SparkSession with different config for different parameters."""
    spark = SparkSession.builder.master(f"local[{cpus}]")
    if loader_type == 's3':
        spark = spark.config('spark.jars.packages',
                             'org.apache.hadoop:hadoop-aws:2.7.6')
    spark = spark \
        .config('spark.driver.memory', get_spark_memory(cpus))
    return spark.getOrCreate()


if __name__ == '__main__':
    with timeit('Total time for `main.py`: {time:.2f} s.'):
        # initialize Spark
        spark = get_spark_session(settings.LOADER, settings.CPU_COUNT)
        sc: SparkContext = spark.sparkContext
        print(f'spark.driver.memory: {sc.getConf().get("spark.driver.memory")}')
        print(f'spark.master: {sc.getConf().get("spark.master")}')

        # initialize loader
        root_path = settings.ROOT_PATH
        loader = LoaderStrategy.get_loader(settings.LOADER)(
            sc=sc, root_path=root_path)
        print(f'Root path in loader: {loader.root_path}')

        # select action
        action = settings.ALGORITHM_ID
        print(f'Action: {action}')
        if action == 1:
            prepare_segment(loader, spark, sc)
        else:
            raise ValueError(f'Action {action} is not recognized.')
