# Movie-Review-Sentiment-Analysis

The basic program is to create a bag of words using the reviews for both the classes. For every tokenized word of the test review document, the program will calculate the relative probability of the word in both the classes and then assign the document to the class with higher probability. This approach gives relatively a good accuracy, but it can be improved by applying natural language processing techniques to the training data.

The improvement strategy to the previous code is to cleanse the data before feeding it to the classifier.

The data collected for the reviews is mostly extracted from the web pages and hence sometimes contain HTML tags along with the data. Therefore, the program starts cleaning the data by initially removing the HTML tags from the input data and later removing the bad characters, punctuations, and symbols. The tokenizer is then used for creating the token out of the input data. 

After tokenizing the words, NLTK’s POS Tagger is used to tag the words and then lemmatize the tagged words. Bag of words are mostly created using only words, but to improve the results, POS tags were used along with the word to create the bag of words. For example, ‘bank’ is stored as bank/N. Stop words are not included in the bag of words. 

Naïve Based Classifier is used to predict the correct class and trained on the training dataset. During the test phase, the classifier generates the probability of the given test document for each class and then assigns the documents to the class (positive or negative) with higher probability. Add-one smoothing is used to eliminate zero probabilities.


