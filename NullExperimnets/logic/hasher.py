from typing import Union

from pyspark.ml.feature import (BucketedRandomProjectionLSH,
                                BucketedRandomProjectionLSHModel)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import settings
from logic.utils import timeit


class Hasher:
    """
    Class for processing `BucketedRandomProjectionLSH` pipeline
    from creating and fitting to ASJ and saving results.
    """
    def __init__(self, spark, loader,
                 filepath: str,
                 bucket_length: float, num_hash_tables: int,
                 input_column: str = 'FEATURES',
                 output_column: str = 'HASHES'):
        self.spark = spark
        self.loader = loader

        self.bucket_length = bucket_length
        self.num_hash_tables = num_hash_tables
        self.input_column = input_column
        self.output_column = output_column

        self.model: Union[BucketedRandomProjectionLSHModel, None] = None

        self.filepath = filepath
        # TODO: move creating to loader
        # filename = self.loader.get_path(filepath, self.get_path())
        # Path(filename).mkdir(parents=True, exist_ok=True)

    def get_path(self) -> str:
        """Return default folder name for model and data storage."""

        return f'bL_{self.bucket_length}__hT_{self.num_hash_tables}'

    @timeit('Load model from file: {time:.2f} s.')
    def load(self):
        """
        Load hash model from file and check if it has excpected properties.
        """

        filename = self.loader.get_path(self.filepath, self.get_path(), 'model.md')
        # self.model = BucketedRandomProjectionLSHModel.load(filename)
        self.model = BucketedRandomProjectionLSHModel.load(r"D:\My_Google_Disk\University\BiQuants\tech\DEV\hash\bL_0.0001__hT_1\model.md")

        props = ['bucketLength', 'numHashTables', 'inputCol', 'outputCol']
        strs = [f'{prop}: {self.model.getOrDefault(prop)}' for prop in props]
        print(f'Model: {" ".join(strs)}')

        assert self.model.getOrDefault('bucketLength') == self.bucket_length
        assert self.model.getOrDefault('numHashTables') == self.num_hash_tables
        assert self.model.getOrDefault('inputCol') == self.input_column
        assert self.model.getOrDefault('outputCol') == self.output_column

    def fit(self, df: DataFrame):
        brp = BucketedRandomProjectionLSH(
            seed=1564,
            bucketLength=self.bucket_length,
            numHashTables=self.num_hash_tables,
            inputCol=self.input_column,
            outputCol=self.output_column,
        )
        self.model = brp.fit(df)

    @timeit('Transform: {time:.2f} s.')
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform dataframe, i.e. generate hashes for elements in dataframe.
        """

        if not self.model:
            raise AttributeError(
                f'Call `fit` or `load` before calling `transform`.')
        return self.model.transform(df)

    def save(self):
        """Save model to file."""

        filename = self.loader.get_path(self.filepath, self.get_path(), 'model.md')
        self.model.write().overwrite().save(filename)

    def __format_results(self, df: DataFrame, distance_column: str) -> DataFrame:
        """Select columns for result dataframe."""
        return df.select(
            # (F.lit(settings.TASK_ID)).alias('TASK_ID'),
            # F.col("datasetA.ROW_ID").alias("ROW_ID"),
            F.col("datasetA.PERSON_ID").alias("PERSON_ID"),
            # F.col("datasetA.HOUSEHOLD_ID").alias("HOUSEHOLD_ID"),
            F.col("datasetB.PERSON_ID").alias("idB"),
            F.col(distance_column)
        )

    def approx_similarity_join(self, df_a: DataFrame, df_b: DataFrame,
                               n: int,
                               threshold: float,
                               distance_column: str = 'DISTANCE') -> DataFrame:
        """
        LSH approximate similarity join: two dataframes are joined on the
        same hashes. Return limited results, if `n` is sent.
        """
        if not self.model:
            raise AttributeError(
                f'Call `fit` or `load` before calling `approx_similarity_join`.')

        result = self.model.approxSimilarityJoin(df_a, df_b, threshold, distance_column)
        result.show()
        result = self.__format_results(result, distance_column)\
            .sort(distance_column)
        result.show()

            # .dropDuplicates(["ROW_ID"])\
            # .sort(distance_column)
        if n:
            result = result.limit(n)
        return result

    def save_results(self, result: DataFrame, filename: str = None,
                     file_type: str = 'csv'):
        """Save results in `csv` or `parquet` format."""

        if not filename:
            filename = self.loader.get_path(
                self.filepath, self.get_path(),
                'results.parquet' if file_type == 'parquet' else 'results.csv')
        with timeit('Write transformed df to file: {time:.2f} s.'):
            if file_type == 'csv':
                result.coalesce(1).write.format("csv").mode("overwrite")\
                    .option("sep", "|").save(filename, header='true')
            else:
                result.write.format("parquet").mode("overwrite").save(filename, header='true')
