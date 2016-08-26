from __future__ import print_function
import argparse
from pyspark import SparkContext
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



def loadStopWords(path): # load the stopword file path # stop word are command words donot give information
    return {s.lower() for s in sc.textFile(path).collect()} # since sw.txt is a small file, use collect to be one file and low it and return a set {}, look up set is very fast 
def tokenizer(s,stopwords=None): # stop words in the tokenizer from "sw.txt" ~700 words, from multiple files 
    if stopwords:
        return [x for x in s.lower().split() if x not in stopwords]  # only return the non-stop words
    else:
        return [x for x in s.lower().split()]

def naive_bayes_train(xRDD,yRDD): # maybe pass a tokenizer for filtering on line 18
    #dRDD=xRDD.zip(yRDD)	
    xRDD=xRDD.zipWithIndex().map(lambda x: (x[1],x[0]))
    yRDD=yRDD.zipWithIndex().map(lambda x: (x[1],x[0]))
    dRDD=xRDD.join(yRDD).map(lambda x: x[1])
    #print("here"*80)
#(wordCountByCatRDD, catCount)=naive_bayes_train(trainingDatasetDocsRDD,trainingDatasetLabelsRDD)
    def documentProcessor(x):
        (document,labels) = x
        cleanLabels = [label for label in labels.upper().split(',') if label in {'MCAT','CCAT','ECAT','GCAT'}] # need to remove the no label documents
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
    catCountRDD = DcuratedRDD.flatMap(lambda x :[(tag,1) for tag in x[1]]).reduceByKey(lambda x,y:x+y)# -> [("c", c_num) , ("m",m_num), ~~~, ~~~]
    catCount = dict(catCountRDD.collect())
    totalWordsByCat = wordCountByCatRDD.map(lambda x: x[1]).reduce(catAccumulator)
    return wordCountByCatRDD,catCount,totalWordsByCat




def naive_bayes_predict (testRDD,wordCountByCatRDD, catCount, totalWordsByCat):

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

sc = SparkContext("local[*]","Naive Bayes", pyFiles = ['nb.py'])
X = sc.textFile(args['x'])
Y = sc.textFile(args['y'])

#load stop words only if a pass is provided
stopwords = None
if args['stop']:
    stopwords = loadStopWords(args['stop'])

(wordCountByCatRDD,catCount,totalWordsByCat) = naive_bayes_train(X,Y)

testRDD = sc.textFile(args['xs']).zipWithIndex() # shift + $ to the end; shift+6 to the beginnig

predictionsRDD = naive_bayes_predict(testRDD,wordCountByCatRDD, catCount, totalWordsByCat)
	
if args['ys'] is not None:
    testLabelsRDD = sc.textFile(args['ys']).zipWithIndex().map(lambda x:(x[1],x[0]))# docId being the first element
    accuracy = score(predictionsRDD, testLabelsRDD)
    print(accuracy*100)
