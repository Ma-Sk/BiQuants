from typing import List

from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.types import StringType, StructField, StructType, FloatType

import settings
from logic.hasher import Hasher
from logic.utils import timeit
from logic.vectorizator import Vectorizator
import math
from pyspark.sql.functions import udf


def distance(one, other):
    summa = 0.0
    for d in range(len(one)):
        delta = one[d] - other[d]
        summa += delta * delta
    return math.sqrt(summa)

def nullcheck(loader, spark, sc: SparkContext):
    """
    Vectorize and hash segment, run LSH to find ANN, check stats(if needed).
    """

    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 100 * 1024 * 1024)

    input_path = settings.INPUT_DATA_PATH
    tech_path = settings.TECH_PATH
    segment_path = settings.SEGMENT_PATH
    output_path = settings.OUTPUT_PATH
    filepath = loader.join(tech_path, 'hash')

    hasher = Hasher(spark, loader, filepath, bucket_length=0.0001,
                    num_hash_tables=1)

    # segment preparation
    filename = loader.get_path(input_path, "seg_col_names.txt")
    columns_seg: List[str] = sc.textFile(filename).collect()

    with timeit('Read segment from file: {time:.2f} s.'):
        columns = [StructField(column, StringType(), True) for column in
                   columns_seg]
        schema = StructType(columns)
        df = spark.read.csv("segmentData/vector1 — копия", header='false', schema=schema,
                            sep="|")
        # df = spark.read.csv("segmentData/EXPERIAN_CONSUMER_VIEW_SEGMENT_143", header='false', schema=schema,
        #                     sep="|")
    df.show()
    hasher.load()
    vectorizator = Vectorizator(spark, loader, tech_path, sparse=False)
    vectorizator.load()
    print('Loaded vectorization model')
    selected_columns = ['PERSON_ID', 'FEATURES']
    df = vectorizator.transform(df, selected_columns,
                                    '2_df_vectorized_not_sparse.parquet')

    filename = loader.get_path(input_path, "seg_col_names.txt")
    columns_seg: List[str] = sc.textFile(filename).collect()
    filename = loader.get_path(segment_path)
    print(filename)
    with timeit('Read segment from file: {time:.2f} s.'):
        columns = [StructField(column, StringType(), True) for column in
                   columns_seg]
        schema = StructType(columns)
        df_seg = spark.read.csv("segmentData/vector2 — копия", header='false', schema=schema,
                                sep="|")
        # df_seg = spark.read.csv("segmentData/EXPERIAN_CONSUMER_VIEW_SEGMENT_146.gz", header='false', schema=schema,
        #                     sep="|")

    df_seg.show()

    n_segment = df_seg.count()
    print(f'n segment: {n_segment}.')

    df_seg = vectorizator.transform(df_seg, selected_columns,
                                    '2_segment_vectorized_not_sparse.parquet')
    filename = loader.get_path(output_path)

    if n_segment < 1 * 10 ** 4:
        threshold = 1 * 10 ** 4
    else:
        threshold = 12

    df.show()
    df_t = hasher.transform(df)
    df_t.show()

    df_seg.show()
    df_t2 = hasher.transform(df_seg)
    df_t2.show()


    spark.udf.register("dist", distance)
    squared_udf = udf(distance)

    nz_pairs_distance = df.crossJoin(df_seg).withColumn("ABS_DISTANCE", squared_udf(df.FEATURES, df_seg.FEATURES))
    nz_pairs_distance.show(20)

