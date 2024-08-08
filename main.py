import numpy as np
from collections import defaultdict
import time

from pyspark.sql import SparkSession
from pyspark.mllib.feature import HashingTF, IDF
from pyspark import SparkConf

config = SparkConf()

spark = SparkSession.builder \
    .config(conf=config) \
    .appName('Assignment') \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

m = 128
p = 4
b = m // p


# Read csv file
df = spark.read.csv("/news.csv", header=True)

# Keeps only the summary coloumn
df = df.select('summary')

# Tokenize the summary coloumn
df = df.rdd.map(lambda x: x.summary)

# Drop columns with value "No summary available."
df = df.filter(lambda x: x != "No summary available.")

# Tokenize the summary coloumn
df = df.map(lambda x: x.split(" "))



##### HashingTF #####
vocab_size = 262144
hashingTF = HashingTF(numFeatures=vocab_size)
tf = hashingTF.transform(df)
tf.cache()

# IDF
idf = IDF().fit(tf)
tfidf = idf.transform(tf)




##### Computing SimHash #####
start = time.time()

# Create a random projection matrix
R = np.random.choice([1, -1], size=(vocab_size, m))
bR = sc.broadcast(R)

# Compute the SimHash of the documents
simhash = tfidf.map(lambda x: x.dot(bR.value) > 0)

# Substitute True with 1 and False with 0
simhash = simhash.map(lambda x: np.array([int(i) for i in x]))



##### LSH #####

def create_pair_dict(x):
    d = defaultdict(set)
    for band, doc in x:
        d[band].add(doc)
    return d

candidate_pairs = ( 
    # Splitting the simhash array into p bands of b elements each
    simhash.map(lambda x: np.split(x, p, axis=0)) 
    # Converting each subarray band into an integer
    .map(lambda x: [int("".join([str(i) for i in band]), 2) for band in x]) 
    # Adding an index to each list of integers
    .zipWithIndex() 
    # Flattening the list of lists and creating key-value pairs where the key is the index of the band and the value is a tuple of the integer hash and the index of the document
    .flatMap(lambda x: [(i, [(el, x[1])]) for i, el in enumerate(x[0])]) 
    # Reducing the key-value pairs by associating to each band a list of tuples where the first element is the integer hash and the second element is the index of the document
    .reduceByKey(lambda x, y: x + y) 
    # Converting the values of each band into a dictionary {band hash: [doc1, doc2, ...]}
    .map(lambda x: create_pair_dict(x[1])) 
    # Flattening the dictionary 
    .flatMap(lambda x: x.items()) 
    # Filtering out lists with more than one element
    .filter(lambda x: len(x[1]) > 1) 
    # Converting into a tuple (doc1, doc2, ...) (I need an hashable object to use the distinct method)
    .map(lambda x: tuple(x[1])) 
    # Removing duplicate tuples
    .distinct()
)

pairs = candidate_pairs.collect()

end = time.time()
time_candidate_pairs = end - start


# Check similarity of candidate pairs
docs = simhash.collect()
start = time.time()
num_candidate_pairs = 0
num_real_pairs = 0

for pair in pairs:
    for i in range(len(pair)):
        for j in range(i+1, len(pair)):
            num_candidate_pairs += 1

            # Count the number of equal bits
            equal_bits = np.sum(docs[pair[i]] == docs[pair[j]])

            if equal_bits >= m - p + 1:
                num_real_pairs += 1
                # print(f"Similarity between doc {pair[i]} and doc {pair[j]}: {equal_bits / m * 100:.2f}%")

end = time.time()
time_real_pairs = end - start

print(f'Num Candidate Pairs: {num_candidate_pairs}')     
print(f'Num Real Pairs: {num_real_pairs}')   

print(f"Time to find candidate pairs: {time_candidate_pairs} seconds")
print(f"Time to find real pairs: {time_real_pairs} seconds")    