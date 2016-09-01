For CSCI8360_F2016 project1: (Dr Shannon Quinn, UGA)

Title: Preforming massive documentation classcifications by Naive Bayes (NB) method through pySpark2.0

Authorship:
Team ChickenBurger:
 	Bahaaeddin Alaila
	   Mojtaba Fazli
	 Shengming Zhang(contact to the authors for a license or bugs) 

Emails: 
	bma09868@uga.edu
	 ms47188@uga.edu
	shzhang1@uga.edu

Date: Aug.30th 2016 

What is it (Introduction): 

The nb.py will preform massive documentations classcification by Naive Bayes method through Spark.
The requirments and the test files can be found in the attaeched project1.pdf file.

How it works? 
Why we choose NB classifiers? Because NB classifiers are good for spam filtering which is based on different feastures, which is very similar to our project. We choose NB method also because it is ease to write up,and it is good for large data set. We applied Multinomial Naive Baye method, by assuming the independence of p(xi|cj), where xi is the word in a document d, and the cj is the jth category. The idea is that  MAP Category = argmax P(cj)*Π(P(xi|cj); where P(cj) is:  (docu. number in Cj)/(total docu. number); Π(P(xi|cj) = p(x1|cj)*p(x2|cj)*...*p(xn|cj), where xi is the word in this document and Cj is the word's category. The reason we can do that is due to the above independent conditional assumption. Therefore, to preform our prediction, we need to: 
1) calculate the prior p(cj) by counting the documents; 
2) calculate the p(xi|cj) for each word and the p(xi|cj) is count(xi, cj)/sum(count xi, cj), which calculation is a little bit tricky.

Details of this project can be found at the documentations in the code. To overcome p(xi|cj)=0 problem, we used Laplace smoothing method (A.K.A add 1 method). We also take log() of the probabilities to overcome the underflow problem for large data set.

Bugs? (or limitations and future improvements)
It does what it should do, implements the NB method to classify documents, we acchieved above 95% accuracy.However, it is very "Naive", to improve our program, we can add more weight to some important identification words, such as momey for ECAT etc. 

Enviroment:
1) This code is written on VIM from a Ubuntu14.04 machine.
2) This code is tested and run on a 2.0 spark from a Ubuntu14.04.
3) The evaluation is made by submitting the resulting label file to the AutoLab online auto grador at https://autolab.cs.uga.edu.

How to run:
1) to get a resulting labeled file:
spark-submit nb.py -x "Training-document-file" -y "Training-label-file" -xs "Test-document-file" -ys "Test-label-file" -o "labeled-file" -stop "stopword-file"

Note: -stop "stopwords-file-path" for better stop words choice
 
2) to get a score for testing proposes:
spark-submit nb.py -x "Training-document-file" -y "Training-label-file" -xs "Test-document-file" -ys "Test-label-file" -o "labeled-file" -stop "stopword-file"
Note:  -o " " for output-labeled-file (which is the result given for testing propose) 

Release:
You can use it for study objective and without ask our permition, but you can NOT use it for you 8360 projects.
Please email us @uga.edu if you found a bug.

ChickenBurger Team
Fall2016 UGA
