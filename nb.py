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
#wordCountByCat.foreach(print)

testDocument = u"Thursday night in the Oval Office with"
testDocument2 = u"Friday Morning out of the Square Swimming Pool without"
testRDD = sc.parallelize([testDocument,testDocument2]).zipWithIndex() # (testDocument, 0)
testRDDSplit=testRDD.flatMap(lambda x: [(w,x[1]) for w in  x[0].lower().split(' ')])
#testRDDSplit.foreach(print)

jointRDD = testRDDSplit.join(wordCountByCat)     #(uga,0).join((uga,{})) => (uga, (0,{})) 
#wordCountByCat.join(testRDDSplit)     #(uga,{}).join((uga,0)) => (uga, ({},0))

#jointRDD.foreach(print)

print("----------")
#what we have is (uga, (0, {M:1}))
#what we need is (0, (uga,{M:1}))
docIDFirstRDD=jointRDD.map(lambda x: (x[1][0], (x[0],x[1][1])))
#docIDFirstRDD.foreach(print)

docRDD = docIDFirstRDD.groupByKey().map(lambda x: (x[0],list(x[1])))
#docRDD.foreach(print)

def labelSplitter(x):
    return [(tag,1) for tag in x.upper().split(',') if tag in {'MCAT','CCAT','ECAT','GCAT'}]
Ysplit = Y.flatMap(labelSplitter)
Yreduced= Ysplit.reduceByKey(lambda x,y: x+y)

#Yreduced.foreach(print)

catCount = dict(Yreduced.collect())

def sumDictValues(d):
    s = 0
    for i in d:
        s += d[i]
    return s

totalNumberOfDocs = sumDictValues(catCount)
print(totalNumberOfDocs)

#print(catCount)

totalNumberOfDocsBroadCast=sc.broadcast(totalNumberOfDocs)
catCountBroadCast=sc.broadcast(catCount)

def naiveBayes(x): #(docID, [(word1, {}), (word2, {}),....])
    maxP = 0
    maxCat = u'MCAT'

    catCount = catCountBroadCast.value
    totalNumberOfDocs=totalNumberOfDocsBroadCast.value

    for cat in catCount:
        p = catCount[cat] / float(totalNumberOfDocs)
        for word in x[1]:
            p *= word[1].get(cat,10e-7) / float(sumDictValues(word[1]))
        if p >= maxP:
            maxP = p
            maxCat = cat
    return (x[0],(maxCat,maxP))



predictionsRDD = docRDD.map(naiveBayes)

predictionsRDD.foreach(print)
import sys

sys.exit(0)





def splitter(x):
    return [(w,1) for w in x.lower().split(' ')] #better tokenizer + stopwords removal
Xsplit = X.flatMap(splitter) #not map!
Xreduced = Xsplit.reduceByKey(lambda x,y: x+y)

VocabularySize = Xreduced.count()
#print(VocabularySize)

