install.packages("lattice")
install.packages("ggplot2")
install.packages("scales")
install.packages("vctrs")
install.packages("class")
install.packages("tree")
install.packages("MLmetrics")
install.packages("randomForest")
install.packages("xgboost")
install.packages("gbm")
install.packages("adabag")
install.packages("rpart")

library(adabag)
library(xgboost)
library(randomForest)
library(readr)
library(lattice)
library(ggplot2)
library(caret)
library(Amelia)
library(pROC)
library(class)
library(tree)
library(MLmetrics)
library(dplyr)
library(gbm)

#carregando teste e treino
teste <- read.csv("test.csv", stringsAsFactors = FALSE)

treino <- read.csv("treino.csv", stringsAsFactors = FALSE)
treino$X <- NULL
#teste <- read.csv("teste.csv", stringsAsFactors = FALSE)
#teste$X <- NULL

teste[is.na(teste)] <- 0
any(is.na(teste))

treino_novo80 <- read.csv("treino_novo80.csv", stringsAsFactors = FALSE)
treino_novo80$X <- NULL
teste_novo20 <- read.csv("teste_novo20.csv", stringsAsFactors = FALSE)
teste_novo20$X <- NULL

# Adicionando novas features no treino

treino["feature1"] <- (treino$platform_name+treino$song_bpm)/2
treino["feature2"] <- (treino$song_bpm+treino$song_gain)/2
treino["feature3"] <- (treino$album_duration+treino$album_tracks)/2
treino["feature4"] <- (treino$artist_fans+treino$artist_albums)/2
treino["feature5"] <- (treino$media_duration+treino$context_type)/2

# Adicionando novas features no teste

teste["feature1"] <- (teste$platform_name+teste$song_bpm)/2
teste["feature2"] <- (teste$song_bpm+teste$song_gain)/2
teste["feature3"] <- (teste$album_duration+teste$album_tracks)/2
teste["feature4"] <- (teste$artist_fans+teste$artist_albums)/2
teste["feature5"] <- (teste$media_duration+teste$context_type)/2


#Adicionando novas features no treino_novo80

treino_novo80["feature1"] <- (treino_novo80$platform_name+treino_novo80$song_bpm)/2
treino_novo80["feature2"] <- (treino_novo80$song_bpm+treino_novo80$song_gain)/2
treino_novo80["feature3"] <- (treino_novo80$album_duration+treino_novo80$album_tracks)/2
treino_novo80["feature4"] <- (treino_novo80$artist_fans+treino_novo80$artist_albums)/2
treino_novo80["feature5"] <- (treino_novo80$media_duration+treino_novo80$context_type)/2


#Adicionando novas features no teste_novo20

teste_novo20["feature1"] <- (teste_novo20$platform_name+teste_novo20$song_bpm)/2
teste_novo20["feature2"] <- (teste_novo20$song_bpm+teste_novo20$song_gain)/2
teste_novo20["feature3"] <- (teste_novo20$album_duration+teste_novo20$album_tracks)/2
teste_novo20["feature4"] <- (teste_novo20$artist_fans+teste_novo20$artist_albums)/2
teste_novo20["feature5"] <- (teste_novo20$media_duration+teste_novo20$context_type)/2


any(is.na(treino)) #procurando NA em treino
any(is.na(teste)) #procurando NA em teste

#limpando e salvando novos conjuntos de dados

#treino <- na.omit(treino)
#teste <- na.omit(teste)


#definindo a semente
set.seed(123)

#row.names(treino) <- NULL
#row.names(teste) <- NULL

#rows.to.delete <- 0.4 * nrow(treino)
#treino <- treino[1:rows.to.delete, ] #treino com 40% da base 

#separando em 80-20 conjunto treino_limpo com 2M de linhas 
# 80% - 1.825.482
# 20% - 456.370



#separando conjunto de treino total com 2M de linhas em treino e teste 

#vetor80 <- sample(1:1825482, size = 1825482, replace = FALSE)
#vetor20 <- sample(1825483:2281852, size = 456370, replace = FALSE)


#treino_novo80 <- treino[+vetor80,] #adicionando novo data frame
#teste_novo20 <- treino[+vetor20,]

any(is.na(treino_novo80))
any(is.na(teste_novo20))

#salvando arquivos manipulados
#write.csv(treino_novo80, "treino_novo80.csv")
#write.csv(teste_novo20, "teste_novo20.csv")

#salvando base manipulada
#write.csv(treino, "treino.csv")
#write.csv(teste, "teste.csv")
#==================================================
#XGBoosting

#dataset fotmato de matriz e atributos e classes separados
set.seed(123)

str(treino_novo80)
str(teste_novo20)
treino_novo80$is_listened <- as.integer(treino_novo80$is_listened)
teste_novo20$is_listened <- as.integer(teste_novo20$is_listened)

#preparando os dados

aux_treino <- as.numeric(treino_novo80[, 14])
aux_treino <- ifelse(aux_treino == 1, 1, 0)
aux_teste <- as.numeric(teste_novo20[, 14])
aux_teste <- ifelse(aux_teste == 1, 1, 0)
aux_teste <- sapply(aux_teste, as.factor)
treino_novo80 <- as.matrix(treino_novo80[, -14])
teste_novo20 <- as.matrix(teste_novo20[, -14])


#treinando

#taxa de aprendizagem eta = 0.1
mod_xgboost <- xgboost(data = treino_novo80, label = aux_treino, max.depth = 9, eta = 0.4, nthread = 5, nrounds = 220, objective = "binary:logistic")

testeXGBoost <- teste
sample_id <- testeXGBoost$sample_id
testeXGBoost$sample_id <- NULL
testeXGBoost <- as.matrix(testeXGBoost)


#teste valendo
classificacao_xgboost <- predict(mod_xgboost, testeXGBoost)

outputXGBoost <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_xgboost)
write.csv(outputXGBoost, "outputXGBoost.csv",row.names = FALSE)



