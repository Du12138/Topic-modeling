import pandas as pd
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from pprint import pprint
import numpy as np
import nltk
nltk.download('wordnet')

data = pd.read_csv('/Users/fdu12/OneDrive/Desktop/df.csv', error_bad_lines=False);
data_text = data[['clean_content']]
data_text['index'] = data_text.index
documents = data_text
print(len(documents))
print(documents[:5])

stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[documents['index'] == 100].values[0][0]
 
print('clean_content')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['clean_content'].map(preprocess)
processed_docs[:10]

dictionary = gensim.corpora.Dictionary(processed_docs)
for i in range(10):
    print(i,dictionary[i])
print('number of total words:',len(dictionary))

for i in range(10):
    print(i,dictionary.dfs[i])
    
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
len(dictionary)
print('number of total words:',len(dictionary))

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(processed_docs[0])
print(bow_corpus[0])

bow_doc_0 = bow_corpus[0]
for i in range(len(bow_doc_0)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_0[i][0], 
                                               dictionary[bow_doc_0[i][0]], 
bow_doc_0[i][1]))

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
    pprint(doc)
    break
    
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=1000, id2word=dictionary, passes=2, workers=4)
 
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
