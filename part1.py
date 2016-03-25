__author__ = 'tracy'

### Things to run when use pyspark on pycharm locally
### Comment them when run on EMR or pyspark
import os, sys
os.environ['SPARK_HOME']= "/Users/tracy/msan-ml/spark-1.6.0-bin-hadoop2.6"
sys.path.append("/Users/tracy/msan-ml/spark-1.6.0-bin-hadoop2.6/python/")
from pyspark import SparkContext
sc = SparkContext()


books = sc.textFile("/Users/tracy/msan-ml/hw2/BX-CSV-Dump/BX-Books.csv")
users = sc.textFile("/Users/tracy/msan-ml/hw2/BX-CSV-Dump/BX-Users.csv")
ratings = sc.textFile("/Users/tracy/msan-ml/hw2/BX-CSV-Dump/BX-Book-Ratings.csv")

noheader_ratings = ratings.filter(lambda x: "User-ID" not in x).map(lambda x: x.split(";"))
noheader_users = users.filter(lambda x: "User-ID" not in x).map(lambda x: x.split(";"))
noheader_books = books.filter(lambda x: "ISBN" not in x).map(lambda x: x.split(";"))

def parse_books(pieces):
    ISBN = pieces[0]
    book_title = pieces[1]
    book_author = pieces[2]
    year_publication = pieces[3]
    publisher = pieces[4]
    image_s = pieces[5]
    image_m = pieces[6]
    image_l = pieces[7]
    return {"ISBN": ISBN, "book_title": book_title, "book_author": book_author,
            "year_publication": year_publication, "publisher": publisher, "image_s": image_s,
            "image_m": image_m, "image_l": image_l}

def parse_ratings(pieces):
    user_id = pieces[0].encode('utf-8')
    ISBN = pieces[1].encode('utf-8')
    book_rating = float(pieces[2].strip('"'))
    return {"user_id": user_id, "ISBN": ISBN, "book_rating": book_rating}

def parse_users(pieces):
    user_id = pieces[0]
    location = pieces[1]
    age = pieces[2].strip('"')
    if age != "NULL":
        age = int(age)
    return {"user_id": user_id, "location": location, "age": age}

parsed_books = noheader_books.map(parse_books)
parsed_ratings = noheader_ratings.map(parse_ratings)
parsed_users = noheader_users.filter(lambda x: len(x)==3).map(parse_users)

print "*********** q1 ***********"

number = parsed_ratings.count()
ratings_counts = parsed_ratings.map(lambda x: x["book_rating"]).countByValue()

print "The number of ratings is 1149780."
print "The result of each rating counts is this:"
print "{0.0: 716109, 1.0: 1770, 2.0: 2759, 3.0: 5996, 4.0: 8904, 5.0: 50974, 6.0: 36924, 7.0: 76457, 8.0: 103736, 9.0: 67541, 10.0: 78610}"
print "The number of implicit is 716109 and the others are explicit."

print "*********** q2 ***********"
print "{0.0: 716109, 1.0: 1770, 2.0: 2759, 3.0: 5996, 4.0: 8904, 5.0: 50974, 6.0: 36924, 7.0: 76457, 8.0: 103736, 9.0: 67541, 10.0: 78610}"


print "*********** q3 ***********"
users_location = parsed_users.map(lambda x: (x["user_id"], x["location"]))
users_ratings = parsed_ratings.map(lambda x: (x["user_id"], x["book_rating"]))
users_location_ratings = users_location.join(users_ratings)
location_ratings = users_location_ratings.map(lambda x: x[1])
sum_count = location_ratings.combineByKey(lambda value: (value, 1),
                                          lambda x, value: (x[0] + value, x[1] + 1),
                                          lambda x, y: (x[0] + y[0], x[1] + y[1]))
averageByKey = sum_count.map(lambda (label, (value_sum, count)): (label, value_sum / count))

print "average ratings per city"
print averageByKey.collectAsMap()

print "*********** q4 ***********"
rating_count = sum_count.map(lambda x: (x[0], x[1][1]))
first_count = rating_count.takeOrdered(1, lambda (k, v): -v)
print "city with more ratings"
print first_count


print "*********** q5 ***********"
counts_per_user = users_ratings.map(lambda (k, v): k).countByValue()
print "number of ratings per user"
print counts_per_user
ISBN_ratings = parsed_ratings.map(lambda x: (x["ISBN"], x["book_rating"]))
ISBN_author = parsed_books.map(lambda x: (x["ISBN"], x["book_author"]))
author_ratings = ISBN_author.join(ISBN_ratings).map(lambda (x,(k,v)): (k, v)).countByKey()
print "number of ratings per author"
print author_ratings

