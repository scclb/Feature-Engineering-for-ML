# Create date: 2020-10-27
# Author: Scc_hy
# reference : Feature Engineering for Machine Learning Principles and Techniques for Data Scientists by Alice Zheng, Amanda Casari
# Tip: Flattening, Filtering, and Chunking


# Bag of word: 记入词频
# Bag-of-n-Grams： n-gram is a sequence of n tokens.
## The larger n is, the richer the information, and the greater the cost.
## Computing n-grams
# ===========================================
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer

# Load the first 10,000 reviews
f = open('')
js = []
for i in range(10000):
  js.append(json.loads(f.readline())

f.close()
review_df = pd.DataFrame(js)

bow_converter = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
bigram_converter = CountVectorizer(ngram_range=(2,2),
                                   token_pattern='(?u)\\b\\w+\\b')
trigram_converter = CountVectorizer(ngram_range=(3,3),
                                   token_pattern='(?u)\\b\\w+\\b')

bow_converter.fit(review_df['text'])
bigram_converter.fit(review_df['text'])
trigram_converter.fit(review_df['text'])

words = bow_converter.get_feature_names()
bigrams = bigram_converter.get_feature_names()
trigrams = trigram_converter.get_feature_names()


# Filtering for Cleaner Features
# ===========================================
## Stopwords
### 直接NLTK 下载， mltk.download()获取

## Frequency-Based Filtering



