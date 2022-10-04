# Cargamos los datos
setwd('/home/jose/Desktop/Universidad/5º Curso/Primer cuatrimestre/Métodos Bayesianos/Primer Proyecto')

tweets <- read.csv('tweets.csv', header=T, sep=',', dec='.')

# Cargamos la librería necesaria para Data Cleaning y NLP
library(tm)

# Como los tweets están ordenados por candidato, barajamos los datos aleatoriamente
set.seed(42)
tweets <- tweets[sample(nrow(tweets)),]

# Juntamos todos los tweets en un corpus
corpus <- Corpus(VectorSource(tweets$text))
inspect(corpus[1:3])

# Aplicando técnicas de Data Cleaning eliminamos los links de cada tweet
removeURL <- content_transformer(function(x) gsub('(f|ht)tp(s?)://\\S+', '', x, perl=T))
clean_corpus <- tm_map(corpus, removeURL)
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, ' ', x))
clean_corpus <- tm_map(clean_corpus, toSpace, '/')
clean_corpus <- tm_map(clean_corpus, toSpace, '@')
clean_corpus <- tm_map(clean_corpus, toSpace, '\\|')
content(clean_corpus[[1]])
clean_corpus <- tm_map(clean_corpus, tolower)
inspect(clean_corpus[1:3])

# A continuación suprimimos números y puntuación
clean_corpus <- tm_map(clean_corpus, removeNumbers)
clean_corpus <- tm_map(clean_corpus, removePunctuation)
inspect(clean_corpus[1:3])

# Asimismo, quitamos las 'stop-words'
clean_corpus <- tm_map(clean_corpus, removeWords,
stopwords('en'))

# Por último, los espacios en blanco
clean_corpus <- tm_map(clean_corpus, stripWhitespace)
inspect(clean_corpus[1:4])
inspect(clean_corpus[1:20])

# Para realizar las wordclouds y modelización de Machine Learning en Python guardamos todo en un csv
dataframe <- data.frame(text=sapply(clean_corpus, identity), 
    stringsAsFactors=F)
tweets$clean_text <- dataframe$text
write.csv(tweets,'data_tweets.csv', row.names=FALSE)

# Identificamos los autores de cada tweet
trump_indices <- which(tweets$handle == 'realDonaldTrump')
clinton_indices <- which(tweets$handle == 'HillaryClinton')

# Dividimos los datos en el set de entrenamiento y de testeo
nobs <- dim(tweets)[1]
train <- 1:round(nobs*0.75)
test <- (round(nobs*0.75) + 1):nobs
twt_train <- tweets[train,]
twt_test <- tweets[test,]
corpus_train <- clean_corpus[train]
corpus_test <- clean_corpus[test]

# Planteamos el BoW creando una matriz de palabras y sus ocurrencias por tweet
twt_dtm <- DocumentTermMatrix(clean_corpus)
inspect(twt_dtm[1:4, 3:10])
twt_dtm_train <- twt_dtm[train,]
twt_dtm_test <- twt_dtm[test,]

# Con esta opción obviamos las palabras que se repiten menos de cinco veces 
five_times_words <- findFreqTerms(twt_dtm_train, 5)
length(five_times_words)
twt_dtm_train <- DocumentTermMatrix(corpus_train, control=list(dictionary=five_times_words))
twt_dtm_test <- DocumentTermMatrix(corpus_test, control=list(dictionary=five_times_words))

# Analizamos con la siguiente función si una palabra está contenida en un tweet o no
convert_count <- function(x) {
    y <- ifelse(x > 0, 1, 0)
    y <- factor(y, levels=c(0, 1), labels=c('No', 'Si'))
}

# Aplicamos dicha función a los sets de entrenamiento del clasificador y testeo de su rendimiento
twt_dtm_train <- apply(twt_dtm_train, 2, convert_count)
twt_dtm_train[1:4, 30:35]
twt_dtm_test <- apply(twt_dtm_test, 2, convert_count)
twt_dtm_test[1:4, 3:10]

# A continuación cargamos la librería necesaria para crear nuestro clasificador
library(e1071)

# El clasificador aprende de los datos de entrenamiento
classifier <- naiveBayes(twt_dtm_train, twt_train$handle)

# Estudiamos su rendimiento con datos de testeo
predicciones <- predict(classifier, newdata=twt_dtm_test)

# Presentamos una tabla con los resultados
table(predicciones, twt_test$handle)

# Adoptamos una perspectiva Bayesiana mediante el 'suavizado' de Laplace
B.clas <- naiveBayes(twt_dtm_train, twt_train$handle,laplace=1)
B.preds <- predict(B.clas, newdata=twt_dtm_test)
table(B.preds, twt_test$handle)