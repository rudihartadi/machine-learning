import pandas as pd
import numpy as np

dataset = pd.read_csv('clean_dataset_stem.csv', sep=';')

'''print(dataset.shape)
print(dataset.head(10))'''

dataset_feature = dataset['ProcessedText'].astype(str)
dataset_label = dataset['Sentimen']
print(dataset_feature)
print(dataset_label)

## Cek Distribusi Label
import matplotlib.pyplot as plt
import seaborn as sns

#visualisasi
'''plt.figure(figsize=(12,8))
sns.displot(dataset_label, label=f'target, skew: {dataset_label.skew():.2f}')
plt.legend(loc='best')
plt.show()'''
print(dataset_label.value_counts())

## BOW
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dataset_feature)
print(vectorizer.get_feature_names_out())

print(x.toarray())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))

x2 = vectorizer2.fit_transform(dataset_feature)
print(vectorizer2.get_feature_names_out())

print(x2.toarray())

## Simulasi Corpus
corpus = [
    'This is the first Document',
    'This document is the second document',
    'And this is the third one.',
    'Is this the first document?'
]
vectorizer3 = CountVectorizer()
z = vectorizer3.fit_transform(corpus)
print(vectorizer3.get_feature_names_out())
print(z.toarray())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2,2))
z2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names_out())
print(z2.toarray())
print(z2)