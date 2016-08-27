from __future__ import print_function
import argparse
from pyspark import SparkContext, SparkConf
#from nltk import word_tokenize

# Defining arguments for parsing the input fies
parser = argparse.ArgumentParser()
parser.add_argument("-x", required = True,
	help="Training set input file x")
parser.add_argument("-y", required = True,
	help="Training set input file Y")
parser.add_argument("-xs", required = True,
	help="Testing set input file x")
parser.add_argument("-ys", required = False,
	help="Testing set input file Y")
parser.add_argument("-o", required = False,
	help="Testing set output file Y")

parser.add_argument("-stop", required = False,
        help="Stopwords list file")

args = vars(parser.parse_args())

# To check if the testing lable or output path are not imported by the user
if args['ys'] is None and args['o'] is None:
    raise ValueError('Either Testing set input file or output file should be provided')



def loadStopWords(path): # load the stopword file path
    # stop word are common words that are not indicative of the topic of the document; "the", "a", "on" ..etc
    # stop words in "sw.txt" ~700 words, compiled from multiple sources 
    # mainly http://ranks.nl/stopwords
    # since sw.txt is a small file, use collect to be one file and low it and return a set {}, look up set is very fast 
    return {s.lower() for s in sc.textFile(path).collect()}

def tokenizer(s,stopwords=None): 
    if stopwords:
        return [x for x in s.lower().split() if x not in stopwords]  # only return the non-stop words
    else:
        return [x for x in s.lower().split()]

def naive_bayes_train(xRDD, yRDD, stopwords=None): # maybe pass a tokenizer for filtering on line 18
    #dRDD=xRDD.zip(yRDD)	#buggy 

    #.zip workaround
    xRDD=xRDD.zipWithIndex().map(lambda x: (x[1],x[0]))
    yRDD=yRDD.zipWithIndex().map(lambda x: (x[1],x[0]))
    dRDD=xRDD.join(yRDD).map(lambda x: x[1])
    
    stopwordsBroadCast=sc.broadcast(stopwords) # stopwordsBroadCast is a handler or say data wrapper

    def documentProcessor(x):
        (document,labels) = x
        # only retain the 4 CATs
        cleanLabels = [label for label in labels.upper().split(',') if label in {u'MCAT',u'CCAT',u'ECAT',u'GCAT'}] 
        if len(cleanLabels) == 0: #has none of the 4 CATs
            return (None,None) #mark them with None to be removed
        # whatever coming back from the tokenizer, distribute the labels on them
        # eg: ("uga represents",["MCAT","CCAT"]) => [("uga",{'MCAT':1,'CCAT':1}), ('represents',{'MCAT':1,'CCAT':1})]
        cleanWords = [(w,{label:1 for label in cleanLabels}) for w in tokenizer(document, stopwords=stopwordsBroadCast.value)] 
        return (cleanWords,cleanLabels)

    def catAccumulator(x,y):
        "joins two dicts together by augmenting respective keys"
        "eg: catAccumulate({'M':1,'C':5},{'C':2,'E':3}) => {'M':1,'C':7,'E':3}"
        for key in y:
            if key in x:
                x[key]+= y[key]
            else:
                x[key]=y[key]
        return x
    DcuratedRDD = dRDD.map(documentProcessor).filter(lambda x: x[1] is not None) #only keep the ones that are not marked None
    wordCountByCatRDD = DcuratedRDD.flatMap(lambda x: x[0]) #spill all the words with their counts per category together
    wordCountByCatRDD = wordCountByCatRDD.reduceByKey(catAccumulator) # reduce by the words to gather the counts for each word
    #catCountRdd: counting how many documents (including duplicates) per category, basic word count done on labels
    catCountRDD = DcuratedRDD.flatMap(lambda x :[(tag,1) for tag in x[1]]).reduceByKey(lambda x,y:x+y)# -> [("c", c_num) , ("m",m_num), ~~~, ~~~]
    catCount = dict(catCountRDD.collect())  #since we only have 4 CATs, we can collect them locally, and broadcast them to the workers
    #totalWordsByCat: counting how many words (not distinct only) per category for later use in naive bayes.
    totalWordsByCat = wordCountByCatRDD.map(lambda x: x[1]).reduce(catAccumulator)
    return wordCountByCatRDD,catCount,totalWordsByCat




def naive_bayes_predict (testRDD,wordCountByCatRDD, catCount, totalWordsByCat, stopwords=None):
    stopwords=sc.broadcast(stopwords)
    testRDDSplit=testRDD.flatMap(lambda x: [(w,x[1]) for w in  tokenizer(x[0],stopwords = stopwordsBroadCast.value)])
    #testRDDSplit.foreach(print)

    jointRDD = testRDDSplit.join(wordCountByCatRDD)  #(uga,0).join((uga,{})) => (uga, (0,{})) 
    #wordCountByCat.join(testRDDSplit)     #(uga,{}).join((uga,0)) => (uga, ({},0))

    #what we have is (uga, (0, {M:1}))
    #what we need is (0, (uga,{M:1}))
    docIDFirstRDD=jointRDD.map(lambda x: (x[1][0], (x[0],x[1][1])))

    docRDD = docIDFirstRDD.groupByKey().map(lambda x: (x[0],list(x[1])))



    def sumDictValues(d):
        s = 0
        for i in d:
            s += d[i]
        return s

    totalNumberOfDocs = sumDictValues(catCount)
    vocabCount = wordCountByCatRDD.count()

    totalWordsByCatBroadCast = sc.broadcast(totalWordsByCat)
    vocabCountBroadCast = sc.broadcast(vocabCount)
    totalNumberOfDocsBroadCast=sc.broadcast(totalNumberOfDocs)
    catCountBroadCast=sc.broadcast(catCount)

    def naiveBayes(x): #(docID, [(word1, {}), (word2, {}),....])
        from math import log
        maxP = float('-inf')
        maxCat = u'MCAT'

        vocabCount = vocabCountBroadCast.value
        catCount = catCountBroadCast.value
        totalNumberOfDocs=totalNumberOfDocsBroadCast.value
        totalWordsByCat = totalWordsByCatBroadCast.value

        for cat in catCount:
            p = log((catCount[cat] + (1.0/len(catCount)))) - log(float(totalNumberOfDocs + 1))
            for word in x[1]:
                p += log(word[1].get(cat,0)+ (1.0/vocabCount)) - log( (float(totalWordsByCat[cat]) + 1))
            if p >= maxP:
                maxP = p
                maxCat = cat
        return (x[0],(maxCat,maxP))



    predictionsRDD = docRDD.map(naiveBayes)
    return predictionsRDD

def score (predictionsRDD,testLabelsRDD):
    joinRDD = predictionsRDD.join(testLabelsRDD)     
    correctNum = 0
    total      = 0
    gradeRDD = joinRDD.map( lambda x : 1 if x[1][0][0] in x[1][1].split(",") else 0 ) # return value after :  
    correctNum = gradeRDD.reduce(lambda x,y :x+y) 
    total = gradeRDD.count()
    return correctNum / float (total)

sc = SparkContext(conf = SparkConf().setAppName("ChickenBurger-NaiveBayes"))
X = sc.textFile(args['x'])
Y = sc.textFile(args['y'])

#load stop words only if a pass is provided
stopwords = None
if args['stop']:
    stopwords = loadStopWords(args['stop'])

(wordCountByCatRDD,catCount,totalWordsByCat) = naive_bayes_train(X,Y,stopwords)

testRDD = sc.textFile(args['xs']).zipWithIndex() # shift + $ to the end; shift+6 to the beginnig

predictionsRDD = naive_bayes_predict(testRDD,wordCountByCatRDD, catCount, totalWordsByCat,stopwords)
	
if args['ys'] is not None:
    testLabelsRDD = sc.textFile(args['ys']).zipWithIndex().map(lambda x:(x[1],x[0]))# docId being the first element
    accuracy = score(predictionsRDD, testLabelsRDD)
    print(accuracy*100)
if args['o'] is not None:
    #sort by docID, extract the predicted labels only, put all the data in one partition, and save to disk (1 partition => 1 file)
    predictionsRDD.sortByKey().map(lambda x: x[1][0]).coalesce(1,False).saveAsTextFile(args['o'])
