from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Data Pipeline") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.read.csv("../DATA/train.csv", header=True)
df = df.drop('UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF')