__author__ = 'tracy'

### Things to run when use pyspark on pycharm locally
### Comment them when run on EMR or pyspark
import os, sys
os.environ['SPARK_HOME']= "/Users/tracy/msan-ml/spark-1.6.0-bin-hadoop2.6"
sys.path.append("/Users/tracy/msan-ml/spark-1.6.0-bin-hadoop2.6/python/")
from pyspark import SparkContext
sc = SparkContext()
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
######################


### Things to be set when use on EMR
AWS_ACCESS_KEY_ID = "AKIAJZAI6CKGGI6JLCXA"
AWS_SECRET_ACCESS_KEY = "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"
sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)



#### Train the naive bayes by calculating the TFIDF for each word in the whole dataset
def filter_word(text):
    return [word for word in text if len(word) >= 3]

# Comment one of them
# This command is for running on local
documents_RDD = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/train_pos.txt")
# This command is for running on EMR connecting to S3
# documents_RDD = sc.textFile("s3n://aml-aml/train_pos.txt")

documents = documents_RDD.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower())\
    .map(lambda line: line.split(" "))\
    .map(lambda x: filter_word(x))\
    .map(lambda x: (1.0, x))

# Comment one of them
# This command is for running on local
documents_neg_RDD = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/train_neg.txt")
# This command is for running on EMR connecting to S3
# documents_neg_RDD = sc.textFile("s3n://aml-aml/train_neg.txt")

documents_neg = documents_neg_RDD.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower())\
    .map(lambda line: line.split(" "))\
    .map(lambda x: filter_word(x))\
    .map(lambda x: (0.0, x))


documents_train = documents.union(documents_neg)

labels = documents_train.map(lambda x: x[0])
train_set = documents_train.map(lambda x: x[1])

hashingTF = HashingTF()
tf = hashingTF.transform(train_set)

tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# Create a labeled point with a positive label and a dense feature vector
training = labels.zip(tfidf).map(lambda x: LabeledPoint(x[0], x[1]))

model = NaiveBayes.train(training)

######### Calculate TFIDF with test data ########

### test_pos data ###
documents_t_RDD = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/test_pos.txt")
# This command is for running on EMR connecting to S3
# documents_RDD = sc.textFile("s3n://aml-aml/test_pos.txt")

documents_t = documents_t_RDD.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower())\
    .map(lambda line: line.split(" "))\
    .map(lambda x: filter_word(x))\
    .map(lambda x: (1.0, x))

documents_neg_t_RDD = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/test_neg.txt")
# This command is for running on EMR connecting to S3
# documents_RDD = sc.textFile("s3n://aml-aml/train_neg.txt")

documents_neg_t = documents_neg_t_RDD.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower())\
    .map(lambda line: line.split(" "))\
    .map(lambda x: filter_word(x))\
    .map(lambda x: (0.0, x))

documents_test = documents_t.union(documents_neg_t)

labels_test = documents_train.map(lambda x: x[0])
test_set = documents_train.map(lambda x: x[1])

hashingTF_t = HashingTF()
tf_t = hashingTF.transform(test_set)

tf_t.cache()
idf_t = IDF(minDocFreq=2).fit(tf_t)
tfidf_t = idf.transform(tf_t)

testing = labels_test.zip(tfidf_t).map(lambda x: LabeledPoint(x[0], x[1]))


######### Make prediction and test accuracy #########
predictionAndLabel = testing.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / testing.count()
print "The accuracy is"
print accuracy
















