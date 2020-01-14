import numpy as np
import nltk
import sklearn
import operator
import requests
import random
import pickle
import itertools
from nltk.probability import FreqDist,ConditionalFreqDist
from sklearn.model_selection import train_test_split
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

nltk.download('stopwords') 
nltk.download('punkt') 
nltk.download('wordnet') 

dataset_neg=open(imdb_train_neg).readlines()
dataset_pos=open(imdb_train_pos).readlines()

dataset_full=[]
for pos_review in dataset_pos:
  dataset_full.append((pos_review,1))
for neg_review in dataset_neg:
  dataset_full.append((neg_review,0))

size_dataset_full=len(dataset_full)
size_test=int(round(size_dataset_full*0.2,0))

list_test_indices=random.sample(range(size_dataset_full), size_test)
train_set=[]
test_set=[]
for i,example in enumerate(dataset_full):
  if i in list_test_indices: test_set.append(example)
  else: train_set.append(example)

random.shuffle(train_set)
random.shuffle(test_set)
print ("Size dataset full: "+str(size_dataset_full))
print ("Size training set: "+str(len(train_set)))
print ("Size test set: "+str(len(test_set)))


lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
# Function taken from Session 1
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

# Function taken from Session 2
def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text


# Functions slightly modified from Session 2

def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
  dict_word_frequency={}
  for instance in training_set:
    sentence_tokens=get_list_tokens(instance[0])
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
  vocabulary=[]
  for word,frequency in sorted_list:
    vocabulary.append(word)
  return vocabulary 
  ##
def train_svm_classifier(training_set, vocabulary): # Function for training our svm classifier
  X_train=[]
  Y_train=[]
  for instance in training_set:
    vector_instance=get_vector_text(vocabulary,instance[0])
    X_train.append(vector_instance)
    Y_train.append(instance[1])
  # Finally, we train the SVM classifier 
  svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
  svm_clf.fit(np.asarray(X_train),np.asarray(Y_train))
  return svm_clf

    vocabulary=get_vocabulary(train_set, 1000)
    svm_clf=train_svm_classifier(new_train_set, vocabulary)


def bag_of_words(words):
  return dict([(word,True)for word in words])

def bigram(words,score_fn=BigramAssocMeasures.chi_sq,n=1000):
  bigram_finder=BigramCollocationFinder.from_words(words)
  bigrams=bigram_finder.nbest(score_fn,n)
  return bag_of_words

def bigram_words(words,score_fn=BigramAssocMeasures.chi_sq,n=1000):
  tuple_words=[]
  for i in words:
    temp=(i,)
    tuple_words.append(temp)
  bigram_finder=BigramCollocationFinder.from_words(words)
  return bag_of_words(tuple_words+bigrams)

def create_word_scores():
  poswords=pickle.load(open(dataset_pos,'rb'))
  negwords=pickle.load(open(dataset_neg,'rb'))

  poswords=list(itertools.chain(*poswords))
  negwords=list(itertools.chain(*negwords))

  word_fd=FredDist()
  cond_word_fd=ConditionalFreqDist()
  for word in poswords:
    word_fd[word]+=1
    cond_word_fd["pos"][word]+=1
  for word in negwords:
    word_fd[word]+=1
    cond_word_fd["neg"][word]+=1

  pos_word_count=count_word_fd['pos'].N()
  neg_word_count=count_word_fd['neg'].N()
  total_word_count=pos_word_count+neg_word_count

  word_scores={}
  for word,freq in word_fd.items():
    pos_score=BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word],(freq,pos_word_count),
                                         total_word_count)      
    neg_score=BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word],(freq,neg_word_count),
                                         total_word_count)   
    word_scores[word]=pos_score+neg_score

  return word_scores

def create_word_bigram_scores():
    posdata=pickle.load(open(dataset_pos,'rb'))
    negdata=pickle.load(open(dataset_neg,'rb'))
    
    poswords=list(itertools.chain(*posdata))
    negwords=list(itertools.chain(*negdata))
    
    bigram_finder=BigramCollocationFinder.from_words(poswords)
    posBigrams=bigram_finder.nbest(BigramAssocMeasures.chi_sq,5000)
    bigram_finder=BigramCollocationFinder.from_words(negwords)
    negBigrams=bigram_finder.nbest(BigramAssocMeasures.chi_sq,5000)

    pos=poswords+posBigrams
    neg=negwords+negBigrams

    word_fd=FredDist()
    cond_word_fd=ConditionalFreqDist()
    for word in pos:
       word_fd[word]+=1
       cond_word_fd["pos"][word]+=1

        
    for word in neg:
      word_fd[word]+=1
      cond_word_fd["neg"][word]+=1
         

    pos_word_count=count_word_fd['pos'].N()
    neg_word_count=count_word_fd['neg'].N()
    total_word_count=pos_word_count+neg_word_count

    word_scores={}
    for word,freq in word_fd.items():
      pos_score=BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word],(freq,pos_word_count),
                                         total_word_count)      
      neg_score=BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word],(freq,neg_word_count),
                                         total_word_count)   
      word_scores[word]=pos_score+neg_score
    

    return word_scores

def find_best_words(word_scores,number):
  best_vals=sorted(word_scores.item(),key=lambda w_s:w_s[1],reverse=True)[:number]
  best_words=set([w for w,s in best_vals])
 
 return best_words

def best_word_features(words):
  global best_words
  return dict([(word,True)for word in words if word in best_words])

pos_review=[]
neg_review=[]

def load_data():
  global pos_review,neg_review
  pos_review=pickle.load(open(dataset_pos,'rb'))
  neg_review=pickle.load(open(dataset_neg,'rb'))

def pos_features(feature_extraction_method):
  posFeatures=[]
  for i in pos_review:
    poswords=[feature_extraction_method(i),'pos']
    posFeatures.append(poswords)
  return posFeatures

def neg_features(feature_extraction_method):
  negFeatures=[]
  for j in neg_review:
    negwords=[feature_extraction_method(i),'neg']
    negFeatures.append(poswords)
  return negFeatures
  train_set=posFeatures[500]+negFeatures[500]
  test_set=posFeatures[500]+negFeatures[500]

def score(LinearSVC):
  svc=LinearSVC()
  svc.train(train_set)
  svc.fit(train_set,test_set)
  pred=svc.predict(test_set)

  return  accuracy_score(train,pred)





















