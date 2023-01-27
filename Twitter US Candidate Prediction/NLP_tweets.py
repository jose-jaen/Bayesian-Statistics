import pandas as pd
import numpy as np
from os import path
from PIL import Image
import turicreate as tc
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline

# Leemos los datos
df = pd.read_csv('data_tweets.csv')

# Creamos el corpus
corpus = [str(i) for i in df['text']]

# Seleccionamos las stop-words
stopwords = set(STOPWORDS)

# Seleccionamos una 'máscara' para dar forma a la nube de palabras
wave_mask = np.array(Image.open('gop2.jpg'))
wordcloud = WordCloud(stopwords=stopwords,
                      background_color='white', mask=wave_mask).generate(corpus)

# Mostramos el resultado final
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
wordcloud.to_file('Trump_wordcloud.png')

# Identificamos a Hillary y Trump
hillary = df['text'][df['handle'] == 'HillaryClinton']
trump = df[['text']][df['handle'] == 'realDonaldTrump']

# Desagregamos por autor
tweets_by_author = df.groupby(['handle'])['text'].apply(list)
text = ' '.join(tweets_by_author[1])

# Seleccionamos stop-words
stopwords = set(STOPWORDS)

# Nueva 'máscara' del Partido Demócrata para Hillary Clinton
wave_mask = np.array(Image.open('dem.jpg'))
wordcloud = WordCloud(stopwords=stopwords,
                      background_color='white', mask=wave_mask).generate(text)

# Resultado final
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Pasamos los datos a SFrame
tweets = tc.SFrame('python.csv')

# Vectorizador de TF-IDF para cada autor de tweets
tweets['tf_idf'] = tc.text_analytics.tf_idf(tweets['clean_text'])
tweets['class'] = tweets['handle'].apply(
    lambda x:1 if x == 'HillaryClinton' else 0)

# Partición de los datos
train_data, test_data = tweets.random_split(0.75, seed=42)

# Creamos clasificador con regresión logística
ml_model = tc.logistic_classifier.create(train_data,
                                         target='class', features=['tf_idf'],
                                         l1_penalty=10, l2_penalty=1000,
                                         validation_set=None)

# Evaluamos nuestro modelo
ml_model.evaluate(test_data)

# Hyperparameter tunning
values = np.logspace(1, 7, num=5)
targets = test_data['class']
f1_scores = []
for l1, l2 in zip(values, values):
        ml_model = tc.logistic_classifier.create(train_data, 
                                                 target='class', features=['tf_idf'],
                                                 l1_penalty=l1, l2_penalty=l2,
                                                 validation_set=None)
        predictions = ml_model.predict(test_data)
        score = tc.evaluation.f1_score(targets, predictions)
        f1_scores.append(score)

print(max(f1_scores))
