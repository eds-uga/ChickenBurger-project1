from __future__ import print_function
from pyspark import SparkContext
sc = SparkContext("local[*]","Naive Bayes", pyFiles = ['nb.py'])

X = sc.textFile('data/X_train_vsmall.txt')
Y = sc.textFile('data/y_train_vsmall.txt')

def splitter(x):
    return [(w,1) for w in x.lower().split(' ')] #better tokenizer + stopwords removal
Xsplit = X.flatMap(splitter) #not map!
Xreduced = Xsplit.reduceByKey(lambda x,y: x+y)

Xreduced.foreach(print)
VocabularySize = Xreduced.count()
print(VocabularySize)
