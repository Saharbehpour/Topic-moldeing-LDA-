#!/usr/bin/env python
# coding: utf-8

# In[11]:


import nltk
# nltk.download('stopwords')
import gensim
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import codecs
import os
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import pyLDAvis.gensim
from pprint import pprint
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd


# In[12]:


documents_list = []
titles=[]
with codecs.open(os.path.join('C:/Users/sss0287/Desktop/spring2020/dr_ting_paper/data', 'data2.txt') ,"r",encoding = 'utf-8') as fin:
    for line in fin.readlines():
        text = line.strip()
        documents_list.append(text)
print("Total Number of Documents:",len(documents_list))
titles.append( text[0:min(len(text),100)] )


# In[13]:


"""
Input  : docuemnt list
Purpose: preprocess text (tokenize, removing stopwords, and stemming)
Output : preprocessed text
"""
# initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = set(stopwords.words('english'))
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
# list for tokenized documents in loop
texts = []
# loop through document list
for i in documents_list:
   # clean and tokenize document string
   raw = i.lower()
   tokens = tokenizer.tokenize(raw)
   # remove stop words from tokens
   stopped_tokens = [i for i in tokens if not i in en_stop]
   # stem tokens
   stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
   # add tokens to list
   texts.append(stopped_tokens)


# In[14]:


dictionary = corpora.Dictionary(texts)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]


# In[15]:




# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix,
                                           id2word=dictionary,
                                           num_topics=50, 
                                           update_every=1,
                                           passes=1)


# In[16]:


topics = lda_model.show_topics()
print(len(topics))
print(topics)
print(len(lda_model.print_topics()))


# In[17]:


from collections import Counter
from PIL import *
from PIL import Image
topics = lda_model.show_topics(num_topics=50,num_words=10,formatted=False)
data_flat = [w for w_list in texts for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])  
print(df)

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(25, 2, figsize=(20,150), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :],  width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], width=0.2, label='Weights')
    ax.set_ylabel('Word Count')
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i),  fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
#     ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=52, y=1.05)    
plt.show()
plt.savefig('testplot.png')
# Image.open('testplot.png').save('testplot.jpg','JPEG')


# In[18]:


df.to_csv("topicmodeling_Result.csv",index = False)


# In[ ]:





# In[19]:


# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=en_stop,
                  background_color='black',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(5, 2, figsize=(20,50), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# In[20]:


topics = lda_model.show_topics(formatted=False)
print(len(topics))


# In[ ]:





# In[ ]:





# In[ ]:




