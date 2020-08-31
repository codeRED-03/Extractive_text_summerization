import numpy as np
import pandas as pd
import nltk
from scipy import spatial 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

paragraph = """SBI, PNB, HDFC, ICICI,  Debit Card Customers: If you are using a bank debit card which doesn't have EMV (Europay, Mastercard and Visa), then you may have to face a problem during money withdrawal from the ATM after 31st December 2019 as your debit card may be blocked from 1st january 2.
               According to the Reserve Bank of India (RBI) guidelines, all Indian banks need to replace the magnetic debit card of their customers with a new EMV card. This debit card replacement is mandatory as it is aimed at meeting the international payment standards.
               Hence, SBI, PNB, HDFC Bank, ICICI Bank or any other bank customers who are using a magnetic debit card are advised to replace their debit card otherwise they will have to face difficulty in money withdrawal from the ATM. 
               The RBI guidelines say all Indian banks will have to replace all magnetic chip-based debit cards with EMV and PIN-based cards by 31st December 2019. Keeping in view of the continuing online frauds on magnetic stripe cards, the RBI has proposed to deactivate them by 31st December 2019.
               So, all magnetic chip-based debit cards will be deactivated from 1st January 2020 (irrespective of the validity of the existing magnetic SBI debit cards). 
               All banks are sending messages to their customers via various means asking them to replace their magnetic chip-based debit card by a new EMV debit card.
               The SBI warned its debit cardholders through a tweet citing, "Apply now to change your Magnetic Stripe Debit Cards to the more secure EMV Chip and PIN-based SBI Debit card at your home branch by 31st December 2019.
               Safeguard yourself with guaranteed authenticity, greater security for online payments and added security against fraud".
               So, the SBI has made it clear that SBI debit card will be blocked if it is a magnetic card. In fact, the SBI has already started deactivating the SBI cards of those SBI accounts in which PAN or Form 60 is not updated. """

embedding_dict = {}
f = open("glove.6B.50d.txt", 'r', encoding = "utf8")
for line in f:
   values = line.split()
   word = values[0]
   vector = np.asarray(values[1:], "float32")
   embedding_dict[word] = vector
f.close()     

def find_nearest(embedding):
    return sorted(embedding_dict.keys(), key=lambda word: spatial.distance.euclidean(embedding_dict[word], embedding))

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    R = re.sub('[^a-zA-Z]', ' ', sentences[i])
    R = R.lower().split()
    R = [word for word in R if not word in stop]
    R = ' '.join(R)
    corpus.append(R)

sentence_vector = []
for i in corpus:
    if len(i)!=0 :
        v = sum([embedding_dict.get(w, np.zeros((50, ))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((50, ))
    sentence_vector.append(v)
    
sim_matrix = np.zeros([len(sentences),len(sentences)])

for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i!=j:
            sim_matrix[i][j] = cosine_similarity(sentence_vector[i].reshape(1,50), sentence_vector[j].reshape(1,50))[0,0]

nx_graph = nx.from_numpy_array(sim_matrix)
scores = nx.pagerank(nx_graph)   

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])