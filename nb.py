from __future__ import print_function
from pyspark import SparkContext
#from nltk import word_tokenize

sc = SparkContext("local[*]","Naive Bayes", pyFiles = ['nb.py'])

X = sc.textFile('data/X_train_vsmall.txt')
Y = sc.textFile('data/y_train_vsmall.txt')


def naive_bayes_train(xRDD,yRDD): # maybe pass a tokenizer for filtering on line 18
    dRDD=xRDD.zip(yRDD)	
#(wordCountByCatRDD, catCount)=naive_bayes_train(trainingDatasetDocsRDD,trainingDatasetLabelsRDD)
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
    DcuratedRDD = dRDD.map(documentProcessor)
    wordCountByCatRDD = DcuratedRDD.flatMap(lambda x: x[0])#.groupByKey().map(lambda x: (x[0],list(x[1])))
    wordCountByCatRDD = wordCountByCatRDD.reduceByKey(catAccumulator)
    #wordCountByCat.foreach(print)    
    catCountRDD = DcuratedRDD.flatMap(lambda x :[(tag,1) for tag in x[1]]).reduceByKey(lambda x,y:x+y)# -> [("c", c_num) , ("m",m_num), ~~~, ~~~]
    return wordCountByCatRDD,catCountRDD




def naive_bayes_predict (testRDD,wordCountByCatRDD, catCountRDD):

    testRDDSplit=testRDD.flatMap(lambda x: [(w,x[1]) for w in  x[0].lower().split(' ')])
    #testRDDSplit.foreach(print)

    jointRDD = testRDDSplit.join(wordCountByCatRDD)  #(uga,0).join((uga,{})) => (uga, (0,{})) 
    #wordCountByCat.join(testRDDSplit)     #(uga,{}).join((uga,0)) => (uga, ({},0))

    #jointRDD.foreach(print)

    #what we have is (uga, (0, {M:1}))
    #what we need is (0, (uga,{M:1}))
    docIDFirstRDD=jointRDD.map(lambda x: (x[1][0], (x[0],x[1][1])))
    #docIDFirstRDD.foreach(print)

    docRDD = docIDFirstRDD.groupByKey().map(lambda x: (x[0],list(x[1])))
    #docRDD.foreach(print)


    catCount = dict(catCountRDD.collect())

    def sumDictValues(d):
        s = 0
        for i in d:
            s += d[i]
        return s

    totalNumberOfDocs = sumDictValues(catCount)
    #print(totalNumberOfDocs)
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
    return predictionsRDD

(wordCountByCatRDD,catCountRDD) = naive_bayes_train(X,Y)
testDocument = u"Thursday night in the Oval Office with"
testDocument2 = u"Friday Morning out of the Square Swimming Pool without"
testRDD = sc.parallelize([testDocument,testDocument2]).zipWithIndex() # (testDocument, 0)
# predictionsRDD = naive_bayes_predict(testingDatasetDocsRDD, wordCountByCatRDD, catCountRDD)

predictionsRDD = naive_bayes_predict(testRDD,wordCountByCatRDD, catCountRDD)

predictionsRDD.foreach(print)


#accuracy = score(predictionsRDD, testingDatasetLabelsRDD)

