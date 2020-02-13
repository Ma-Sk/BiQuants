import os
from functools import reduce
from typing import List

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, count, col
from pyspark.sql.types import *
from pyspark.sql import DataFrame

# Если hadoop_home уже настроен это трока не нужна
os.environ['HADOOP_HOME'] = r"D:\Programms\Spark\spark-2.4.4-bin-hadoop2.7"

conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

sqlContext = SQLContext(sc)

segmentFile = r"D:\My_Google_Disk\University\BiQuants\data\EXPERIAN_CONSUMER_VIEW_SEGMENT_143"
infoFile = r"D:\My_Google_Disk\University\BiQuants\data\REF_SVOD_EXPERIAN_MEASURES_MAPPING.gz"

segmentHeaderFile = r"D:\My_Google_Disk\University\BiQuants\data\seg_col_names.txt"
infoHeaderFile = r"D:\My_Google_Disk\University\BiQuants\data\measures_mapping_col_names.txt"


def loadMainColumnNames(colNamesFileName: str) -> List:
    """
    get only important column names
    :param: colNamesFileName (str): name of file with column names
    :return: List: list of important column names
    """
    f = open(colNamesFileName, "r")
    res = f.read().splitlines()
    return res[3:]


def loadHeaderNames(headerFileName: str) -> StructType:
    """
    get column names scheme from file
    :param: headerFileName: name of file with column names
    :return: column names scheme
    """
    f = open(headerFileName, "r")
    names = f.read().splitlines()
    res = []
    for x in names:
        res.append(StructField(x, StringType(), True))
    return StructType(res)


def loadDataFrame(dataFileName: str, headerScheme: StructType) -> DataFrame:
    """
    load dataframe from file
    :param dataFileName: file name of data
    :param headerScheme: column names for chosen dataframe
    :return: loaded dataframe
    """
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='false', delimiter='|').load(
        dataFileName, schema=headerScheme)
    return df


def getColumnsWithNull(colNames: List, segmentDF: DataFrame, infoDF: DataFrame) -> str:
    """
    get column names that contain null values(WORK IN PROGRESS)
    :param colNames: names of column that are checked for a null values
    :param segmentDF: dataframe that is checked for a null values
    :param infoDF: dataframe with availiable values in column
    :return: string with all column names with null value
    """
    res = ""
    for x in colNames:
        t1 = segmentDF.select(x).distinct().selectExpr(x + " as vals")

        t2 = infoDF.select("VALUES_SVOD_TEXT").filter(
            infoDF["CONSUMER_VIEW_FIELDS"] == x).distinct().selectExpr("VALUES_SVOD_TEXT as vals")

        if t1.join(t2, ["vals"], 'leftanti').count() == 1:
            res += x + "\n"
    return res


def getAmountOfRecordsWithNull(segmentDF: DataFrame, columnToCheck: List = None) -> int:
    """
    get Amount Of Records With Null value
    :param segmentDF: dataframe that is checked for a null values
    :param columnToCheck: names of column that are checked for a null values
    :return: amount of records with null
    """
    if columnToCheck is None:
        columns = segmentDF.columns
    else:
        columns = columnToCheck

    # print(segmentDF.subtract(segmentDF.dropna()).count())
    return segmentDF.where(reduce(lambda x, y: x | y, (f.col(x).isNull() for x in columns))).count()


def getAmountOfRecordsWithNullForEachColumn(segmentDF: DataFrame, columnToCheck: List = None) -> DataFrame:
    """
    get Amount Of Records With Null value for each column
    :param segmentDF: dataframe that is checked for a null values
    :param columnToCheck: names of column that are checked for a null values
    :return: dataframe with amount of records with null for each column
    """
    if columnToCheck is None:
        columns = segmentDF.columns
    else:
        columns = columnToCheck

    colNulls = segmentDF.select([count(when(col(c).isNull(), c)).alias(c) for c in columns])
    # colNulls.show()
    return colNulls


def getPercentOfRecordsWithNullForEachColumn(segmentDF: DataFrame, columnToCheck: List = None) -> DataFrame:
    """
    get percent Of Records With Null value for each column
    :param segmentDF: dataframe that is checked for a null values
    :param columnToCheck: names of column that are checked for a null values
    :return: dataframe with percent of records with null for each column
    """
    if columnToCheck is None:
        columns = segmentDF.columns
    else:
        columns = columnToCheck

    colNullsProc = segmentDF.select([(count(when(col(c).isNull(), c)) / segmentDF.count()).alias(c) for c in columns])
    return colNullsProc


def getNullDistributionInRow(segmentDF: DataFrame, columnToCheck: List = None) -> DataFrame:
    """
    get null count|amount of rows with that null count dataframe
    :param segmentDF: dataframe that is checked for a null values
    :param columnToCheck: names of column that are checked for a null values
    :return: dataframe with null count|amount of rows with that null count
    """
    if columnToCheck is None:
        columns = segmentDF.columns
    else:
        columns = columnToCheck
    nullCountDF = segmentDF.select(
        sum([segmentDF[col].isNull().cast(IntegerType()) for col in columns]).alias('null_count'))
    return nullCountDF.groupBy('null_count').count().orderBy('null_count')


if __name__ == "__main__":
    fw = open("out.txt", "w")
    colNames = loadMainColumnNames(segmentHeaderFile)
    segmentDF = loadDataFrame(segmentFile, loadHeaderNames(segmentHeaderFile))
    infoDF = loadDataFrame(infoFile, loadHeaderNames(infoHeaderFile))
    nullCount = getAmountOfRecordsWithNull(segmentDF, colNames)
    print(nullCount)
    print(nullCount / segmentDF.count())
    getAmountOfRecordsWithNullForEachColumn(segmentDF, colNames).coalesce(1).write.save("amount", format="csv",
                                                                                        delimiter="|")
    getPercentOfRecordsWithNullForEachColumn(segmentDF, colNames).coalesce(1).write.save("percent", format="csv",
                                                                                         delimiter="|")

    getNullDistributionInRow(segmentDF, colNames).coalesce(1).write.save("SegNullDist", format="csv", delimiter="|")

    res = getColumnsWithNull(colNames, segmentDF, infoDF)
    # fw.write(res)

