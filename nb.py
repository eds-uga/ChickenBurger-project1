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
    cleanWords = [(w,len(cleanLabels)) for w in document.lower().split(' ')]
    return (cleanWords,cleanLabels)

Dcurated = D.map(documentProcessor)
Dcurated.foreach(print)

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
