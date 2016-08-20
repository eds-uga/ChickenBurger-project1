from __future__ import print_function
from pyspark import SparkContext
#from nltk import word_tokenize

sc = SparkContext("local[*]","Naive Bayes", pyFiles = ['nb.py'])

X = sc.textFile('data/X_train_vsmall.txt')
Y = sc.textFile('data/y_train_vsmall.txt')
D=X.zip(Y)

def documentProcessor(x):
    (document,labels) = x
    cleanLabels = [label for label in labels.upper().split(',') if label in {'MCAT','CCAT','ECAT','GCAT'}]
    cleanWords = [(w,{label:1 for label in cleanLabels}) for w in document.lower().split(' ')]
    return (cleanWords,cleanLabels)

def catAccumulator(x,y):
    for key in y:
        if key in x:
            x[key]+= y[key]
        else:
            x[key]=y[key]
    return x
Dcurated = D.map(documentProcessor)
wordCountByCat = Dcurated.flatMap(lambda x: x[0])#.groupByKey().map(lambda x: (x[0],list(x[1])))
wordCountByCat = wordCountByCat.reduceByKey(catAccumulator)
wordCountByCat.foreach(print)

import sys

sys.exit(0)
def splitter(x):
    return [(w,1) for w in x.lower().split(' ')] #better tokenizer + stopwords removal
Xsplit = X.flatMap(splitter) #not map!
Xreduced = Xsplit.reduceByKey(lambda x,y: x+y)

VocabularySize = Xreduced.count()
#print(VocabularySize)

def labelSplitter(x):
    return [(tag,1) for tag in x.upper().split(',') if tag in {'MCAT','CCAT','ECAT','GCAT'}]
Ysplit = Y.flatMap(labelSplitter)
Yreduced= Ysplit.reduceByKey(lambda x,y: x+y)
Yreduced.foreach(print)
