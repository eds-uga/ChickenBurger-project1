from __future__ import print_function # for testing purposes
import argparse 
from pyspark import SparkContext, SparkConf # for spark usage

# Defining arguments for parsing the input files
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
# To check if the testing label file and output path are imported by the user
if args['ys'] is None and args['o'] is None:
    raise ValueError('Either Testing set input file or output file should be provided')

def loadStopWords(path): 
    """
    load the stopword file path
    stop word are common words that are not indicative of the topic of the document; "the", "a", "on" ..etc
    stop words in "sw.txt" ~700 words, compiled from multiple sources 
    mainly http://ranks.nl/stopwords
    the HTML - Special Entity Codes are appended at the end 
    since sw.txt is a small file, use collect to be one file and lowcase it and return as a set {}, because look up set is very fast 
    """
    return {s.lower() for s in sc.textFile(path).collect()}

def tokenizer(s,stopwords=None):
    """
    tokenizer function
    s is the document to be splitted, 
    "stopwords=None" is a placeholder, will be none until the user assign a value for it. 
    This function returns a list of "cleaned" words, "in" requires "iterable type", None is not iterable, set stepwords = or set(), It only
    returns the non-stop words, not-number(float), not "white-space". 
    """ 
    stopwords = stopwords or set()     # an empty set class:len(set()) if not assigned i 
    def isfloat(x):                     
        try:
            float(x)
            return True
        except ValueError:
            return False
        return [x for x in s.lower().split() if x not in stopwords and x != '' and not isfloat(x)]   
# in this project we applied Multinomial Naive Baye method, by assuming the independence of p(xi|cj), where xi is the word in a docu. d, and the cj is the jth category. The core idea is:  Category = argmax P(cj)*Π(P(xi|cj); where P(cj) is:  (docu. number in Cj)/(total docu. number); Π(P(xi|cj) = p(x1|cj)*p(x2|cj)*~~~, where xi is the word in this document and Cj is the word's category. We can do that since the above independent conditional assumption. Therefore, to preform our predict we need to: 1) calcuate the prior p(cj) by counting the docu.s; 2) calculate p(xi|cj) for each word. and p(xi|cj) is count(xi, cj)/sum(count xi, cj), which is a little bit triky. The below fn is to preform the countings
# naive_bayes_train fn 
# take inputs of training doc-file(:x) and label-file(:y) data then calculate the wordCountByCatRDD,catCount,totalWordsByCat

def naive_bayes_train(xRDD, yRDD, stopwords=None): 
    '''
    Inputs : xRDD : training-`docu. file; yRDD: label-file 
    in this project we applied Multinomial Naive Baye method, by assuming the independence of p(xi|cj) for each words 
    where xi is the word in a docu., d, and the cj is the jth category. The main idea is:  Category = argmax P(cj)*Π(P(xi|cj); 
    where P(cj) is:  (docu. number in Cj)/(total docu. number); Π(P(xi|cj) = p(x1|cj)*p(x2|cj)*...*p(xn|cj), 
    where xi is the word in this document and Cj is the word's category. The reason we can do that is based on the above independent conditional assumption. 
    Therefore, to preform our prediction we need to: 
    1) calculate the prior p(cj) by counting the docu.s; 
    2) calculate p(xi|cj) for each word and p(xi|cj) is: count(xi, cj)/sum(count xi, cj), that calculation is a little bit triky. 
    
    This naive_bayes_train fn is to preform the countings based on the training files, (doc. and label files) 
	it takes inputs of training doc-file(:x) and label-file(:y) data then calculate the wordCountByCatRDD,catCount,totalWordsByCat
    '''
    #dRDD=xRDD.zip(yRDD)	#buggy, work around by using zipWithIndex()
    #.zip workaround
    xRDD=xRDD.zipWithIndex().map(lambda x: (x[1],x[0])) # zipWithIndex() -> [{(element0), 0}, {(element1),1}~~~} , label each docu. and return set of tuples
    yRDD=yRDD.zipWithIndex().map(lambda x: (x[1],x[0])) 
    dRDD=xRDD.join(yRDD).map(lambda x: x[1])            #dRDD are table of docu.s and labels
    
    stopwordsBroadCast=sc.broadcast(stopwords)          
  
    def documentProcessor(x):
        (document,labels) = x   #recall x is tuple type, we assign the first part of x as document, second part as labels    
        # only retain the 4 CATs, return a list -> [], u'MCAT' for python string format
        cleanLabels = [label for label in labels.upper().split(',') if label in {u'MCAT',u'CCAT',u'ECAT',u'GCAT'}] 
        if len(cleanLabels) == 0: #has none of the 4 CATs
            return (None,None) #mark them with None to be removed
        
        # w goes through the docus. and returns whatever coming back from the tokenizer(,) and aslo distribute the labels on them to form the cleanwords list
        # eg: ("uga represents",["MCAT","CCAT"]) => [("uga",{'MCAT':1,'CCAT':1}), ('represents',{'MCAT':1,'CCAT':1})]
        cleanWords = [(w,{label:1 for label in cleanLabels}) for w in tokenizer(document, stopwords=stopwordsBroadCast.value)] # set the stopwords
        return (cleanWords,cleanLabels)  # return "cleanworlds" with label and count for 1

    def catAccumulator(x,y):
        "joins two dicts{key,value} together by augmenting respective keys"                  # one line documentation
        "eg: catAccumulate({'M':1,'C':5},{'C':2,'E':3}) => {'M':1,'C':7,'E':3}"
        for key in y:
            if key in x:
                x[key]+= y[key]
            else:
                x[key]=y[key]
        return x
    DcuratedRDD = dRDD.map(documentProcessor).filter(lambda x: x[1] is not None) #only keep the ones that are not marked None
    wordCountByCatRDD = DcuratedRDD.flatMap(lambda x: x[0]) #flatten the x[1] of D_curated
    wordCountByCatRDD = wordCountByCatRDD.reduceByKey(catAccumulator) # reduce by the words to gather the counts for each word's category
    #catCountRdd: counting how many documents (including duplicates) in each category for calculating the prior P(cj)
    catCountRDD = DcuratedRDD.flatMap(lambda x :[(tag,1) for tag in x[1]]).reduceByKey(lambda x,y:x+y)# -> [("c", c_num) , ("m",m_num), ~~~, ~~~]
    catCount = dict(catCountRDD.collect())  #since we only have 4 CATs, we can collect them locally, and broadcast them to the workers, and get ready for calculating the counting for docu.s in each category
    #totalWordsByCat: counting how many words (not distinct only) per category for later use in naive bayes.
    totalWordsByCat = wordCountByCatRDD.map(lambda x: x[1]).reduce(catAccumulator)
    return wordCountByCatRDD,catCount,totalWordsByCat




def naive_bayes_predict (testRDD,wordCountByCatRDD, catCount, totalWordsByCat, stopwords=None):
    "based on word counting to calculate the MAP"
    stopwordsBroadCast=sc.broadcast(stopwords)
    testRDDSplit=testRDD.flatMap(lambda x: [(w,x[1]) for w in tokenizer(x[0],stopwords = stopwordsBroadCast.value)])
    #testRDDSplit.foreach(print)

    jointRDD = testRDDSplit.leftOuterJoin(wordCountByCatRDD)  #(uga,0).join((uga,{})) => (uga, (0,{})) 
    #wordCountByCat.join(testRDDSplit)     #(uga,{}).join((uga,0)) => (uga, ({},0))

    #what we have is (uga, (0, {M:1}))
    #what we need is (0, (uga,{M:1}))
    docIDFirstRDD=jointRDD.map(lambda x: (x[1][0], (x[0],x[1][1])))

    docRDD = docIDFirstRDD.groupByKey()



    def sumDictValues(d):
    	"count the total nomber of documents including the over-counted ones"
        s = 0
        for i in d:
            s += d[i]
        return s

    totalNumberOfDocs = sumDictValues(catCount) 
    vocabCount = wordCountByCatRDD.count()

    totalWordsByCatBroadCast = sc.broadcast(totalWordsByCat) # tell all worker the total words of each category
    vocabCountBroadCast = sc.broadcast(vocabCount)           # each word's count in every category
    totalNumberOfDocsBroadCast=sc.broadcast(totalNumberOfDocs)
    catCountBroadCast=sc.broadcast(catCount)

    def naiveBayes(x): #(docID, [(word1, {}), (word2, {}),....])
    	"""
    	preform the NB method. 
		1) Laplace smoothing method(add 1 method) to overcome the p(xi|cj)=0 problem
		2) taking log() to overcome the underflow problem for large data set 0.01*10^10 -> underflow number
    	"""    	
        from math import log
        maxP = float('-inf')
        maxCat = u'MCAT'

        vocabCount = vocabCountBroadCast.value
        catCount = catCountBroadCast.value
        totalNumberOfDocs=totalNumberOfDocsBroadCast.value
        totalWordsByCat = totalWordsByCatBroadCast.value

        for cat in catCount:
            p = log(catCount[cat] + 1.0) - log(totalNumberOfDocs + len(catCount))
            for word in x[1]:
                wordDict = word[1] or {}
                p += log(wordDict.get(cat,0) + 1.0 ) - log( (float(totalWordsByCat[cat]) + vocabCount))
            if p >= maxP:
                maxP = p
                maxCat = cat
        return (x[0],(maxCat,maxP))



    predictionsRDD = docRDD.map(naiveBayes)
    return predictionsRDD


def score (predictionsRDD,testLabelsRDD):
    "this function gives the score of our prediction based on the provided labeling files for small data set"
    joinRDD = predictionsRDD.join(testLabelsRDD)     
    correctNum = 0
    total      = 0
    gradeRDD = joinRDD.map( lambda x : 1 if x[1][0][0] in x[1][1].split(",") else 0 )
    correctNum = gradeRDD.reduce(lambda x,y :x+y) 
    total = gradeRDD.count()
    return correctNum / float (total)

sc = SparkContext(conf = SparkConf().setAppName("ChickenBurger-NaiveBayes"))
X = sc.textFile(args['x'])
Y = sc.textFile(args['y'])

stopwords = None
if args['stop']:
    stopwords = loadStopWords(args['stop'])

(wordCountByCatRDD,catCount,totalWordsByCat) = naive_bayes_train(X,Y,stopwords)

testRDD = sc.textFile(args['xs']).zipWithIndex()  # add index to "xs"

predictionsRDD = naive_bayes_predict(testRDD,wordCountByCatRDD, catCount, totalWordsByCat,stopwords)
	
if args['ys'] is not None: # calculate score
    testLabelsRDD = sc.textFile(args['ys']).zipWithIndex().map(lambda x:(x[1],x[0]))# docId being the first element
    accuracy = score(predictionsRDD, testLabelsRDD)
    print(accuracy*100)
if args['o'] is not None:
    #sort by docID, extract the predicted labels only, put all the data in one partition, and save to disk (1 partition => 1 file)
    predictionsRDD.sortByKey(ascending = True, numPartitions=1).map(lambda x: x[1][0]).saveAsTextFile(args['o'])
