'''
Created on 18-Feb-2016

@author: ajadhav
'''
from __future__ import division

from pyspark.sql import Row
from pyspark import SparkContext
from pyspark.mllib.linalg import DenseVector
import os
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameReader
from pyspark import SparkContext
from pyspark.sql import SQLContext


os.environ["SPARK_HOME"] = "/home/ajadhav/Documents/spark-1.6.0-bin-hadoop2.6"
sc= SparkContext()
sqlContext = SQLContext(sc)
# Load csv file using Spark-csv in data-frame form
df = sqlContext.read.load('/media/ajadhav/New Volume/Downloads/iris.csv',format='com.databricks.spark.csv',header='true',inferSchema='true') 
df.first()
df.printSchema()

# COnvert categorical column to numeric
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="Species", outputCol="Numeric_Species")
model = stringIndexer.fit(df)
indexed = model.transform(df)
from pyspark.ml.feature import OneHotEncoder, StringIndexer
encoder = OneHotEncoder(inputCol="Numeric_Species", outputCol="Numeric_Species_Vec")
encoded = encoder.transform(indexed)
encoded.show()

# Convert Data to Spark Type (LabeledPoint)
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import DenseVector
parsedData = encoded.map(lambda x: LabeledPoint(x[6], DenseVector(x[1:5])))
training, test = parsedData.randomSplit([0.7, 0.3], seed = 11L)

# Create a Model
from pyspark.mllib.tree import DecisionTree
model = DecisionTree.trainClassifier(training,numClasses=3, categoricalFeaturesInfo={},impurity='gini', maxDepth=30, maxBins=100)

# Predict a model on Test Data
ActualLabels  = test.map(lambda x: x.label)
TestFeatures  = test.map(lambda x: x.features)
PredictedLabels = model.predict(TestFeatures)
PredLabels = ActualLabels.zip(PredictedLabels)
PredLabels_DF = PredLabels.toDF()

# Print the Accuracy of a model on Test Data
Accuracy = PredLabels.filter(lambda (v, p): v == p).count()/PredLabels.count()
print("Accuracy is = " + str(Accuracy))
