# Look alike service

The service resolve approximate nearest neighbour search for two different sets.
It has two options to run on: `prepare_data` pipeline perform preprocessing for 
big, rare changeable `set1` of users, whether `prepare_segment` uses already 
prepared data from `set1` to prepare `set2` in the same way and then join two sets
on hash values, calculate distance and save to s3 a file with results.


##### Tips:

- for debugging Spark jobs you can use Spark UI on default port 4040;
- if you change `*.sh` files under Windows OS, be sure about unix line endings.

##### Helpful links:

- Spark code on GitHub:
    1. LSH: https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/feature/LSH.scala
    2. BucketedRandomProjectionLSH: https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/feature/BucketedRandomProjectionLSH.scala
- PySpark docs for bucket random projection: https://spark.apache.org/docs/latest/ml-features#bucketed-random-projection-for-euclidean-distance
- Paper for bucket random projection: https://mlpack.org/papers/lsh.pdf
