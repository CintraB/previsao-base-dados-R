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
install.packages("caret")
install.packages("rpart.plot")

library(rpart)
library(rpart.plot)
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


teste_novo20 <- read.csv("teste_novo20.csv", stringsAsFactors = FALSE)
teste_novo20$X <- NULL
teste <- read.csv("test.csv",stringsAsFactors = FALSE)
teste[is.na(teste)] <- 0
any(is.na(teste))

#Adicionando novas features no teste_novo20


teste_novo20["feature1"] <- (teste_novo20$listen_type+teste_novo20$context_type)/2
teste_novo20["feature2"] <- (teste_novo20$listen_type+teste_novo20$artist_id)/2
teste_novo20["feature3"] <- (teste_novo20$listen_type+teste_novo20$release_date)/2
teste_novo20["feature4"] <- (teste_novo20$context_type+teste_novo20$media_duration)/2
teste_novo20["feature5"] <- (teste_novo20$context_type+teste_novo20$platform_family)/2

# Adicionando novas features no teste

teste["feature1"] <- (teste$listen_type+teste$context_type)/2
teste["feature2"] <- (teste$listen_type+teste$artist_id)/2
teste["feature3"] <- (teste$listen_type+teste$release_date)/2
teste["feature4"] <- (teste$context_type+teste$media_duration)/2
teste["feature5"] <- (teste$context_type+teste$platform_family)/2


#==================================================
#Ensamble

#lendo arquivos de saidas
saida_validacao_AD <- read.csv("saida_validacao_AD.csv", stringsAsFactors = FALSE)
saida_validacao_GB <- read.csv("saida_validacao_GB.csv", stringsAsFactors = FALSE)
saida_validacao_XGB <- read.csv("saida_validacao_XGB.csv", stringsAsFactors = FALSE)

#mesclando todas saidas dos algoritmos em uma base 
base_especialistas <- saida_validacao_AD
base_especialistas <-  mutate(base_especialistas,saida_validacao_GB)
base_especialistas <- mutate(base_especialistas,saida_validacao_XGB)


base_especialistas$classificacao_arvoreD <- as.integer(base_especialistas$classificacao_arvoreD)
base_especialistas$classificacao_gb <- as.integer(base_especialistas$classificacao_gb)
base_especialistas$validacao <- as.integer(base_especialistas$validacao)

#adicionando saida esperada do conjunto de validacao
base_especialistas <- mutate(base_especialistas,teste_novo20$is_listened)

colnames(base_especialistas) <- c('arvoreD','GB','XGB','is_listened')


write.csv(base_especialistas, "base_especialistas.csv",row.names = FALSE)


#Arvore de decisao


set.seed(123)
#base_especialistas$is_listened <- as.factor(base_especialistas$is_listened)



controle_poda = rpart.control(minsplit = 30, minbucket = 10)
mod_arvore = rpart(is_listened ~ ., data = base_especialistas, control = controle_poda)



barplot(mod_arvore$variable.importance)


#prp(mod_arvore)

#gerando base com saidas do teste de cada algoritmo.

outputArvoreD <- read.csv("outputArvoreD.csv", stringsAsFactors = FALSE)
outputArvoreD$sample_id <- NULL
outputGB <- read.csv("outputGB.csv", stringsAsFactors = FALSE)
outputGB$sample_id <- NULL
outputXGBoost <- read.csv("outputXGBoost.csv", stringsAsFactors = FALSE)
outputXGBoost$sample_id <- NULL

outputArvoreD <- ifelse(outputArvoreD > 0.5,1,0)
outputGB <- ifelse(outputGB > 0.5,1,0)
outputXGBoost <- ifelse(outputXGBoost > 0.5,1,0)

outputArvoreD <- as.data.frame(outputArvoreD)
outputGB <- as.data.frame(outputGB)
outputXGBoost <- as.data.frame(outputXGBoost)


teste_final <- data.frame(arvoreD = outputArvoreD,GB = outputGB, XGB = outputXGBoost)
colnames(teste_final) <- c('arvoreD','GB','XGB')


#teste valendo
classificacao_arvoreD <- predict( mod_arvore,teste_final)
classificacao_arvoreD <- as.data.frame(classificacao_arvoreD)


ArvoreD_Ens <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_arvoreD)
colnames(ArvoreD_Ens) <- c('sample_id','is_listened')

write.csv(ArvoreD_Ens, "ArvoreD_Ens.csv",row.names = FALSE)


#===============================================================================
#testando com o XGBoost para ganhar do resultado de 0.66834 da arvore 

set.seed(123)
#lendo arquivos de saidas
saida_validacao_AD <- read.csv("saida_validacao_AD.csv", stringsAsFactors = FALSE)
saida_validacao_GB <- read.csv("saida_validacao_GB.csv", stringsAsFactors = FALSE)
saida_validacao_XGB <- read.csv("saida_validacao_XGB.csv", stringsAsFactors = FALSE)

#mesclando todas saidas dos algoritmos em uma base 
base_especialistas <- saida_validacao_AD
base_especialistas <-  mutate(base_especialistas,saida_validacao_GB)
base_especialistas <- mutate(base_especialistas,saida_validacao_XGB)


base_especialistas$classificacao_arvoreD <- as.integer(base_especialistas$classificacao_arvoreD)
base_especialistas$classificacao_gb <- as.integer(base_especialistas$classificacao_gb)
base_especialistas$validacao <- as.integer(base_especialistas$validacao)

#adicionando saida esperada do conjunto de validacao
base_especialistas <- mutate(base_especialistas,teste_novo20$is_listened)

colnames(base_especialistas) <- c('arvoreD','GB','XGB','is_listened')





set.seed(123)


base_especialistas$is_listened <- as.integer(base_especialistas$is_listened)
#teste_novo20$is_listened <- as.integer(teste_novo20$is_listened)

#preparando os dados

aux_treino <- as.numeric(base_especialistas[, 4])
aux_treino <- ifelse(aux_treino == 1, 1, 0)
#aux_teste <- as.numeric(teste_novo20[, 14])
#aux_teste <- ifelse(aux_teste == 1, 1, 0)
#aux_teste <- sapply(aux_teste, as.factor)
base_especialistas <- as.matrix(base_especialistas[, -4])
#teste_novo20 <- as.matrix(teste_novo20[, -14])


#treinando

#taxa de aprendizagem / eta = 0.4 depth = 10 rounds = 1500 
mod_xgboost <- xgboost(data = base_especialistas, label = aux_treino, max.depth = 50, eta = 0.005, nthread = 8, nrounds = 1500, objective = "binary:logistic")
#validacao <- predict(mod_xgboost,teste_novo20)



#gerando base com saidas do teste de cada algoritmo.

outputArvoreD <- read.csv("outputArvoreD.csv", stringsAsFactors = FALSE)
outputArvoreD$sample_id <- NULL
outputGB <- read.csv("outputGB.csv", stringsAsFactors = FALSE)
outputGB$sample_id <- NULL
outputXGBoost <- read.csv("outputXGBoost.csv", stringsAsFactors = FALSE)
outputXGBoost$sample_id <- NULL

outputArvoreD <- as.data.frame(outputArvoreD)
outputGB <- as.data.frame(outputGB)
outputXGBoost <- as.data.frame(outputXGBoost)


teste_final <- data.frame(arvoreD = outputArvoreD,GB = outputGB, XGB = outputXGBoost)
colnames(teste_final) <- c('arvoreD','GB','XGB')



testeXGBoost <- teste_final
sample_id <- teste$sample_id
testeXGBoost$sample_id <- NULL
testeXGBoost <- as.matrix(testeXGBoost)


#teste valendo
classificacao_xgboost <- predict(mod_xgboost, testeXGBoost)

XGBoostEns <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_xgboost)
write.csv( XGBoostEns, "XGBoostEns.csv",row.names = FALSE)


#===============================================================================
#testando com o Gradient Boost para ganhar do resultado de 0.66834 da arvore 

#lendo arquivos de saidas
saida_validacao_AD <- read.csv("saida_validacao_AD.csv", stringsAsFactors = FALSE)
saida_validacao_GB <- read.csv("saida_validacao_GB.csv", stringsAsFactors = FALSE)
saida_validacao_XGB <- read.csv("saida_validacao_XGB.csv", stringsAsFactors = FALSE)

#mesclando todas saidas dos algoritmos em uma base 
base_especialistas <- saida_validacao_AD
base_especialistas <-  mutate(base_especialistas,saida_validacao_GB)
base_especialistas <- mutate(base_especialistas,saida_validacao_XGB)


base_especialistas$classificacao_arvoreD <- as.integer(base_especialistas$classificacao_arvoreD)
base_especialistas$classificacao_gb <- as.integer(base_especialistas$classificacao_gb)
base_especialistas$validacao <- as.integer(base_especialistas$validacao)



#adicionando saida esperada do conjunto de validacao
base_especialistas <- mutate(base_especialistas,teste_novo20$is_listened)

colnames(base_especialistas) <- c('arvoreD','GB','XGB','is_listened')






teste_novo20$is_listened <- as.character(teste_novo20$is_listened)
base_especialistas$is_listened <- as.character(base_especialistas$is_listened)


# gbm classe com formato character 2000 trees , 80
mod_gb <- gbm(is_listened ~ ., data = base_especialistas, n.trees = 1000,cv.folds = 15,shrinkage = 0.005, n.minobsinnode = 40,n.cores = 8, distribution = "bernoulli",verbose = TRUE)


#teste valendo
classificacao_gb <- predict.gbm(mod_gb,teste_final,type = "response")
output <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_gb)
write.csv(output, "GBEns.csv",row.names = FALSE)


