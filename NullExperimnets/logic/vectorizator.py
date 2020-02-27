from typing import List, Union

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (OneHotEncoderEstimator, StringIndexer,
                                VectorAssembler)
from pyspark.sql import DataFrame

from logic.utils import timeit


class Vectorizator:

    def __init__(self, spark, loader, filepath: str = None, sparse=False):
        self.spark = spark
        self.loader = loader
        self.sparse = sparse
        self.filepath = filepath
        self.model: Union[PipelineModel, None] = None

    def load(self, filename: str = None):
        if not filename:
            filename = f'vectorization_model_{"" if self.sparse else "not_"}sparse.md'
        file = self.loader.get_path(self.filepath, filename)
        # self.model = PipelineModel.load(file)
        self.model = PipelineModel.load(r"D:\My_Google_Disk\University\BiQuants\tech\DEV\vectorization_model_not_sparse.md")

    def save(self, filename: str = None):
        if not filename:
            filename = f'vectorization_model_{"" if self.sparse else "not_"}sparse.md'
        file = self.loader.get_path(self.filepath, filename)
        self.model.write().overwrite().save(file)

    def __create_stages(self, categorical_columns,
                        string_indexer_prefix,
                        one_hot_prefix,
                        handle_invalid='keep'):
        stages = []
        for col in categorical_columns:
            indexer = StringIndexer(
                inputCol=col,
                outputCol=f'{string_indexer_prefix}_{col}',
                handleInvalid=handle_invalid
            )
            if self.sparse:
                encoder = OneHotEncoderEstimator(
                    inputCols=[indexer.getOutputCol()],
                    outputCols=[f'{one_hot_prefix}_{col}'],
                    handleInvalid=handle_invalid
                )
                stages += [indexer, encoder]
            else:
                stages += [indexer]
        return stages

    def fit(self, df: DataFrame, categorical_columns: List[str],
            string_indexer_prefix: str = 'category',
            one_hot_prefix: str = 'one_hot',
            handle_invalid: str = 'keep',
            ):
        print(f'Columns count: {len(df.columns)}.')
        assert len(categorical_columns) == 59, \
            'Number of `categorical_columns` is incorrect: ' \
            f'{len(categorical_columns)}'
        print(f'Categorical columns count: {len(categorical_columns)}')
        print(categorical_columns)

        stages = self.__create_stages(
            categorical_columns, string_indexer_prefix,
            one_hot_prefix, handle_invalid)

        if self.sparse:
            assembler_inputs = [f'{one_hot_prefix}_{c}'
                                for c in categorical_columns]
        else:
            assembler_inputs = [f'{string_indexer_prefix}_{c}'
                                for c in categorical_columns]

        assembler = VectorAssembler(
            inputCols=assembler_inputs,
            outputCol="FEATURES")

        stages += [assembler]
        pipeline = Pipeline(stages=stages)
        with timeit('Fit pipeline: {time:.2f} s.'):
            self.model = pipeline.fit(df)

    def transform(self, df: DataFrame,
                  selected_columns: List[str], filename: str) -> DataFrame:
        if not self.model:
            raise AttributeError(
                f'Call `fit` or `load` before calling `transform`.')

        result = self.model.transform(df).select(selected_columns)

        file = self.loader.get_path(self.filepath, filename)
        with timeit('Write transformed df to file: {time:.2f} s.'):
            result.write.format("parquet").mode("overwrite").save(file)
        return self.spark.read.parquet(file)
