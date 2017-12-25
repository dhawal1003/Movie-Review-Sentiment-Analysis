from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sys
import re
import string
import glob
import os
import json
import numpy as np
import math
import time

def remove_html_tags(text):
    return TAG_RE.sub(' ', text)

def get_wordnet_tag(tag):
    if tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return 'n'

def fetchBagOfWords(className):
    global TAG_RE, punc, stopWords,lemmatizer, trainingPath, vocabulary, posTagger, trainingFilesCountPerClass
    filePath = trainingPath + "\\" + className + "\\*.txt"
    bagOfWords = {}
    noOfWords = 0
    count = 0
    files = glob.glob(filePath)
    table = str.maketrans({key: None for key in string.punctuation})

    if(os.path.isfile("./"+className+"BigramWords.txt")):
        dictFile = open(className+"BigramWords.txt",'r')
        bagOfWords = json.loads(dictFile.read()) 
        wordsCount = open(className+'BigramCount.txt','r')
        wordsCount = wordsCount.read()
        return bagOfWords, wordsCount
      
    for fle in files:
        count += 1
        if(count == trainingFilesCountPerClass):
            break
        with open(fle, encoding="utf-8") as f:
            for line in f:
                # Remove  html tags 
                line = remove_html_tags(line)
                
                # Remove punctuations
                line = line.translate(table)
        
                # Tokenize
                line = word_tokenize(line)

                # POS Tagging for lemmatization
                tagged_words = nltk.pos_tag(line)

                for index in range(len(tagged_words)-1):
                    word1 = tagged_words[index][0].lower() + "/" + tagged_words[index][1][0]
                    word2 = tagged_words[index+1][0].lower() + "/" + tagged_words[index+1][1][0]
                    token = word1 + " " + word2
                    vocabulary.add(token)
                    noOfWords += 1
                    if token not in bagOfWords:
                        bagOfWords[token] = 1
                    else:
                        bagOfWords[token] += 1
    
    dictFile = open(className+'BigramWords.txt', 'w+')
    dictFile.write(json.dumps(bagOfWords))
    wordsCount = open(className+'BigramCount.txt','w+')
    wordsCount.write(str(noOfWords))

    return bagOfWords, noOfWords

def fetchCorrectClass():
    global TAG_RE, punc, stopWords,lemmatizer, testingPath, vocabularyLength, bagOfWords , totalWordsinClass, matchedDocsCount, testFilesPerClassCount
    classNames = ['pos','neg']
    table = str.maketrans({key: None for key in string.punctuation})
    matchedClassCount = 0
    
    # Checking files in both class folders
    for classIndex in range(len(classNames)):
        count = 0
        perClassCount = 0
        path = testingPath + "\\" + classNames[classIndex] + "\\*.txt"
        files = glob.glob(path)
        for fle in files:
            probabilityPerClass = []
            for i in range(len(classNames)):            
                prob = 0.0
                with open(fle, encoding="utf-8") as f:
                    for line in f:
                        # Remove  html tags 
                        line = remove_html_tags(line)                 
                        # Remove punctuations
                        line = line.translate(table)
                        # Tokenize
                        line = word_tokenize(line)
                        # POS Tagging for lemmatization
                        tagged_words = nltk.pos_tag(line)
                        
                        for index in range(len(tagged_words)-1):
                            word1 = tagged_words[index][0].lower() + "/" + tagged_words[index][1][0]
                            word2 = tagged_words[index+1][0].lower() + "/" + tagged_words[index+1][1][0]
                            token = word1 + " " + word2  
                            if token in bagOfWords[i]:
                                prob += math.log((bagOfWords[i][token]+1)/(int(totalWordsinClass[i])+int(vocabularyLength)))
                            else:
                                prob += math.log((1)/(int(totalWordsinClass[i])+int(vocabularyLength)))
                prob += math.log(0.5)
                probabilityPerClass.append(prob)
            if np.argmax(probabilityPerClass) == classIndex:
                matchedClassCount += 1
                perClassCount += 1
            count += 1
            if(count == testFilesPerClassCount):
                break
        matchedDocsCount.append(perClassCount)
    print("Accuracy on the test dataset is: ",round((matchedClassCount/(testFilesPerClassCount*2))*100,2),"%")
    

def printTime(start):
    end = time.time()
    duration = end - start
    if duration < 60:
        return "Time elapsed: " + str(round(duration, 2)) + "s."
    else:
        mins = int(duration / 60)
        secs = round(duration % 60, 2)
        if mins < 60:
            return "Time elapsed: " + str(mins) + "m " + str(secs) + "s."
        else:
            hours = int(duration / 3600)
            mins = mins % 60
            return "Time elapsed: " + str(hours) + "h " + str(mins) + "m " + str(secs) + "s."

if __name__ == "__main__":
    
    if(len(sys.argv)<2):
        print("Pass all the parameters required. E.g. python classifyBigram.py <Number of test documents(Per class)>")
        exit()
    print("Program is executing. Please wait...")
    print()
    
    start = time.time()
    trainingPath = "E:\\MSCS_StudyMaterial\\Fall 2017\\NLP\\Final Project\\aclImdb\\train"
    testingPath = "E:\\MSCS_StudyMaterial\\Fall 2017\\NLP\\Final Project\\aclImdb\\test"
    punc = list(string.punctuation)
    stopWords = set(stopwords.words('english'))
    vocabulary = set()
    stopWords.remove('not')
    TAG_RE = re.compile(r'<[^>]+>')

    # Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    trainingFilesCountPerClass = 12500
    testFilesPerClassCount = int(sys.argv[1])

    bagOfWords = []
    totalWordsinClass = []
    matchedDocsCount = []

    
    truePositiveCount = 0
    falsePositiveCount = 0
    trueNegativeCount = 0
    falseNegativeCount = 0

    positiveClassPrior = 0.5
    negativeClassPrior = 0.5

    
    positiveBagOfWords, totalPositiveWords = fetchBagOfWords('pos')
    negativeBagOfWords, totalNegativeWords = fetchBagOfWords('neg')

    bagOfWords.append(positiveBagOfWords)
    bagOfWords.append(negativeBagOfWords)

    totalWordsinClass.append(totalPositiveWords)
    totalWordsinClass.append(totalNegativeWords)

    if(os.path.isfile("./vocabBigram.txt")):
        vocabFile = open("./vocabBigram.txt",'r')
        vocabularyLength = vocabFile.read()
    else:
        vocabularyLength = len(vocabulary)
        vocabFile = open("./vocabBigram.txt",'w+')
        vocabFile.write(str(vocabularyLength))
    
    fetchCorrectClass()
    # print(matchedDocsCount)
    truePositiveCount = matchedDocsCount[0]
    trueNegativeCount = matchedDocsCount[1]

    falsePositiveCount = testFilesPerClassCount - trueNegativeCount
    falseNegativeCount = testFilesPerClassCount - truePositiveCount

    # print("TP: ",truePositiveCount)
    # print("TN: ",trueNegativeCount)
    # print("FP: ",falsePositiveCount)
    # print("FN: ",falseNegativeCount)

    precision = (truePositiveCount/(truePositiveCount+falsePositiveCount))
    recall = (truePositiveCount/(truePositiveCount+falseNegativeCount))

    print("Precision: ",round(precision*100,2),"%")
    print("Recall: ",round(recall*100,2),"%")
    print(printTime(start))
    