__author__ = 'tracy'

### Things to run when use pyspark on pycharm locally
### Comment them when run on EMR or pyspark
import os, sys
os.environ['SPARK_HOME']= "/Users/tracy/msan-ml/spark-1.6.0-bin-hadoop2.6"
sys.path.append("/Users/tracy/msan-ml/spark-1.6.0-bin-hadoop2.6/python/")
from pyspark import SparkContext
import numpy as np
import collections
sc = SparkContext()
##############

### Things to be set when use on EMR
AWS_ACCESS_KEY_ID = "AKIAJZAI6CKGGI6JLCXA"
AWS_SECRET_ACCESS_KEY = "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"
sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)


######## Training Part ########

def add_counts(tuple, count_class, v_class):
    w, count = tuple
    return (w, count, count_class, v_class)

#### train pos #####
# train_pos_files = sc.textFile("s3n://aml-aml/train_pos.txt")
train_pos_files = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/train_pos.txt")

train_pos_counts = train_pos_files.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower()) \
        .flatMap(lambda x: x.split()) \
        .filter(lambda x: len(x) == 3)\
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x,y: x+y)

# add count(c) as the third element in tuple
count_pos = train_pos_counts.map(lambda (x, y): y).reduce(lambda x, y: x + y)
v_pos = train_pos_counts.count()

p_w_pos = train_pos_counts.map(lambda x: add_counts(x, count_pos, v_pos)) \
    .map(lambda (w, count, count_pos, v_pos): (w, float((count + 1))/(count_pos + v_pos + 1)))\
    .collectAsMap()

# p_w_pos is dictionary
# {u'reception': 0.0009182736455463728, u'time': 0.0018365472910927456, u'far': 0.0009182736455463728,
# u'serious': 0.0013774104683195593, u'original': 0.0018365472910927456}

##### train neg ####
# train_neg_files = sc.textFile("s3n://aml-aml/train_neg.txt")
train_neg_files = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/train_neg.txt")

train_neg_counts = train_neg_files.map( lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower()) \
        .flatMap(lambda x: x.split()) \
        .filter(lambda x: len(x) == 3)\
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x,y:x+y)

# add count(c) as the third element in tuple
count_neg = train_neg_counts.map(lambda (x, y): y).reduce(lambda x, y: x + y)
v_neg = train_neg_counts.count()

p_w_neg = train_neg_counts.map(lambda x: add_counts(x, count_neg, v_neg)) \
    .map(lambda (w, count, count_neg, v_neg): (w, float((count + 1))/(count_neg + v_neg + 1)))\
    .collectAsMap()


######## Testing Part ########
# M: number of unique words in one document
# ni: number of one word in one document

def add_class_w(test_counts, p_w_pos, p_w_neg, v_pos, v_neg, count_pos, count_neg):
    """

    :param test_counts:
    :param p_w_pos:
    :param p_w_neg:
    :param v_pos:
    :param v_neg:
    :param count_pos:
    :param count_neg:
    :return: p_w_pos is a list of all words' probability; p_w_pos_v one particular word p
    """
    label = test_counts[0]
    word_list = test_counts[1]
    result_row = []
    for item in word_list:
        word, counts = item
        if word in p_w_pos:
            p_w_pos_v = p_w_pos[word]
        else:
            p_w_pos_v = 1.0/(count_pos + v_pos + 1)
        if word in p_w_neg:
            p_w_neg_v = p_w_neg[word]
        else:
            p_w_neg_v = 1.0/(count_neg + v_neg + 1)
        result_row.append((word, counts, p_w_pos_v, p_w_neg_v))
    result = (label, result_row)
    return result


def multiply_counts(classify_data):
    """

    :param classify_data:
    :return: p_w_pos_n is power to probability of one word in this document
    """
    multiply_row = []
    for item in classify_data[1]:
        word, counts, p_w_pos_v, p_w_neg_v = item
        log_p_w_pos_n = np.log(p_w_pos_v)*counts
        log_p_w_neg_n = np.log(p_w_neg_v)*counts
        multiply_row.append((log_p_w_pos_n, log_p_w_neg_n))
    multiply = (classify_data[0], multiply_row)
    return multiply

def classifier(data, P_POS, P_NEG):
    total_pos = 0
    total_neg = 0
    for item in data[1]:
        log_p_w_pos_n, log_p_w_neg_n = item
        total_pos += log_p_w_pos_n
        total_neg += log_p_w_neg_n
    total_pos = total_pos + P_POS
    total_neg = total_neg + P_NEG

    if total_pos > total_neg:
        classificaion = 1.0
    else:
        classificaion = 0.0
    result = (data[0], classificaion)
    return result


P_POS = float(train_pos_files.count())/(train_pos_files.count() + train_neg_files.count())
P_NEG = float(train_neg_files.count())/(train_pos_files.count() + train_neg_files.count())


### test pos ###
# test_pos_counts_file = sc.textFile("s3n://aml-aml/test_pos.txt")
test_pos_counts_file = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/test_pos.txt")

test_pos_counts = test_pos_counts_file.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower()) \
    .map(lambda x: x.split()) \
    .map(lambda x: collections.Counter(x))\
    .map(lambda x: x.items())\
    .map(lambda x: (1.0, x))

classify_data_pos = test_pos_counts.map(lambda x: add_class_w(x, p_w_pos, p_w_neg, v_pos, v_neg,
                                                count_pos, count_neg))

classification_result_pos = classify_data_pos.map(lambda x: multiply_counts(x))\
    .map(lambda x: classifier(x, P_POS, P_NEG))


total_pos = classification_result_pos.map(lambda x: (x, 1)).map(lambda (x, y): y).reduce(lambda x, y: x + y)
wrong_pos = classification_result_pos.map(lambda (x, y): abs(x - y)).sum()


### test neg ###
# test_pos_counts_file = sc.textFile("s3n://aml-aml/test_pos.txt")
test_pos_counts_file = sc.textFile("/Users/tracy/msan-ml/hw2/aclImdb/test_pos.txt")

test_pos_counts = test_pos_counts_file.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').lower()) \
    .map(lambda x: x.split()) \
    .map(lambda x: collections.Counter(x))\
    .map(lambda x: x.items())\
    .map(lambda x: (0.0, x))

classify_data_neg = test_pos_counts.map(lambda x: add_class_w(x, p_w_pos, p_w_neg, v_pos, v_neg,
                                                count_pos, count_neg))

classification_result_neg = classify_data_neg.map(lambda x: multiply_counts(x))\
    .map(lambda x: classifier(x, P_POS, P_NEG))


total_neg = classification_result_neg.map(lambda x: (x, 1)).map(lambda (x, y): y).reduce(lambda x, y: x + y)
wrong_neg = classification_result_neg.map(lambda (x, y): abs(x - y)).sum()

### Calculate Accuracy ###
accuracy = (total_pos + total_neg - wrong_pos - wrong_neg)/(total_pos + total_neg)
print "The accuracy is"
print accuracy













