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
install.packages("e1071")

library(e1071)
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

#treino <- read.csv("train.csv", stringsAsFactors = FALSE)
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

treino["feature1"] <- (treino$listen_type+treino$context_type)/2
treino["feature2"] <- (treino$listen_type+treino$artist_id)/2
treino["feature3"] <- (treino$listen_type+treino$release_date)/2
treino["feature4"] <- (treino$context_type+treino$media_duration)/2
treino["feature5"] <- (treino$context_type+treino$platform_family)/2



# Adicionando novas features no teste

teste["feature1"] <- (teste$listen_type+teste$context_type)/2
teste["feature2"] <- (teste$listen_type+teste$artist_id)/2
teste["feature3"] <- (teste$listen_type+teste$release_date)/2
teste["feature4"] <- (teste$context_type+teste$media_duration)/2
teste["feature5"] <- (teste$context_type+teste$platform_family)/2

#Adicionando novas features no treino_novo80

treino_novo80["feature1"] <- (treino_novo80$listen_type+treino_novo80$context_type)/2
treino_novo80["feature2"] <- (treino_novo80$listen_type+treino_novo80$artist_id)/2
treino_novo80["feature3"] <- (treino_novo80$listen_type+treino_novo80$release_date)/2
treino_novo80["feature4"] <- (treino_novo80$context_type+treino_novo80$media_duration)/2
treino_novo80["feature5"] <- (treino_novo80$context_type+treino_novo80$platform_family)/2


#Adicionando novas features no teste_novo20


teste_novo20["feature1"] <- (teste_novo20$listen_type+teste_novo20$context_type)/2
teste_novo20["feature2"] <- (teste_novo20$listen_type+teste_novo20$artist_id)/2
teste_novo20["feature3"] <- (teste_novo20$listen_type+teste_novo20$release_date)/2
teste_novo20["feature4"] <- (teste_novo20$context_type+teste_novo20$media_duration)/2
teste_novo20["feature5"] <- (teste_novo20$context_type+teste_novo20$platform_family)/2

copia_saida_teste20 <- teste_novo20$is_listened

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

#any(is.na(treino_novo80))
#any(is.na(teste_novo20))

#salvando arquivos manipulados
#write.csv(treino_novo80, "treino_novo80.csv")
#write.csv(teste_novo20, "teste_novo20.csv")

#salvando base manipulada
#write.csv(treino, "treino.csv")
#write.csv(teste, "teste.csv")

#==================================================
#Arvore D. 

set.seed(123)
treino_novo80_copia <- treino_novo80
teste_novo20_copia <- teste_novo20


any(is.na(treino_novo80_copia)) #procurando NA em treino
treino_novo80_copia <- na.omit(treino_novo80_copia)

any(is.na(teste_novo20_copia)) #procurando NA em teste
teste_novo20_copia <- na.omit(teste_novo20_copia)

rows.to.delete <- 1 * nrow(treino_novo80) #coloquei 100%
treino_novo80_copia <- treino_novo80_copia[1:rows.to.delete, ] 


#coloco tudo em factor (os is_listened) pra rodar
#treino_novo80_copia$is_listened <- as.factor(treino_novo80_copia$is_listened)
#teste_novo20_copia$is_listened <- as.factor(teste_novo20_copia$is_listened)




#30 10
controle_poda = rpart.control(minsplit = 30, minbucket = 10)
mod_arvoreD = rpart(is_listened ~ ., data = treino_novo80_copia , control = controle_poda)

prp(mod_arvoreD)

mod_arvoreD$variable.importance




classificacao_arvoreD <- predict(mod_arvoreD, teste_novo20_copia[, -14])
classificacao_arvoreD <- as.data.frame(classificacao_arvoreD)
classificacao_arvoreD <- ifelse(classificacao_arvoreD > 0.5, 1, 0)
classificacao_arvoreD <- as.data.frame(classificacao_arvoreD)
classificacao_arvoreD <- sapply(classificacao_arvoreD,as.factor)

tp1 <- sum((teste_novo20_copia$is_listened == 1) & (classificacao_arvoreD == 1))
fp1 <- sum((teste_novo20_copia$is_listened == 0) & (classificacao_arvoreD == 1))
tn1 <- sum((teste_novo20_copia$is_listened == 0) & (classificacao_arvoreD == 0))
fn1 <- sum((teste_novo20_copia$is_listened == 1) & (classificacao_arvoreD == 0))
matrizc_arvoreD <- matrix(c(tn1, fn1, fp1, tp1), nrow = 2, ncol = 2, dimnames = list(c("0","1"), c("0","1")))

matrizc_arvoreD


#metricas
#precisao XGBoost
precisao_arvoreD = tp1/(tp1+fp1)
precisao_arvoreD

#recall XGBoost
recall_arvoreD = tp1/(tp1+fn1)
recall_arvoreD

precisao_arvoreD <- as.data.frame(precisao_arvoreD)
recall_arvoreD <- as.data.frame(recall_arvoreD)


metricas_arvoreD <- merge.data.frame(precisao_arvoreD,recall_arvoreD)

write.csv(metricas_arvoreD, "Metricas_ArvoreD.csv",row.names = TRUE)




#escrevendo saida da validacao para juntar no ensamble 

saida_validacao_AD <- data.frame(classificacao_arvoreD)
write.csv(saida_validacao_AD,"saida_validacao_AD.csv",row.names = FALSE)


#teste valendo
classificacao_arvoreD <- predict(mod_arvoreD,teste)

outputArvoreD <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_arvoreD)

any(is.na(outputArvoreD)) #procurando NA em treino


write.csv(outputArvoreD,"outputArvoreDsolo.csv",row.names = FALSE)




#==================================================
#Random Forest
set.seed(123)
treino_novo80_copia <- treino_novo80
teste_novo20_copia <- teste_novo20


any(is.na(treino_novo80_copia)) #procurando NA em treino
treino_novo80_copia <- na.omit(treino_novo80_copia)

any(is.na(teste_novo20_copia)) #procurando NA em teste
teste_novo20_copia <- na.omit(teste_novo20_copia)

rows.to.delete <- 1 * nrow(treino_novo80) #coloquei 100%
treino_novo80_copia <- treino_novo80_copia[1:rows.to.delete, ] 


#coloco tudo em factor (os is_listened) pra rodar
treino_novo80_copia$is_listened <- as.factor(treino_novo80_copia$is_listened)
teste_novo20_copia$is_listened <- as.factor(teste_novo20_copia$is_listened)


#treinando #60
mod_rf <- randomForest(formula = is_listened~.,data = treino_novo80_copia, ntree = 60,localImp = TRUE,do.trace = 1)
classificacao_randomForest <- predict(mod_rf,teste_novo20_copia[,-14])

#ajustando e gerando matriz de confusão 
matrizc_randomf <- confusionMatrix(classificacao_randomForest, as.factor(teste_novo20_copia[, 14]), positive = "1", mode = "prec_recall")
matrizc_randomf$table

#metricas
precisao_randomforest <- matrizc_randomf$byClass["Precision"]
recall_randomforest <- matrizc_randomf$byClass["Recall"]
metricas_randomforest <- c(precisao_randomforest, recall_randomforest)
metricas_randomforest
metricas_randomforest <- data.frame(metricas_randomforest)

write.csv(metricas_randomforest, "Metricas_RandomForest.csv",row.names = TRUE)

#escrevendo saida da validacao para juntar no ensamble 

saida_validacao_RF <- data.frame(classificacao_randomForest)
write.csv(saida_validacao_RF,"saida_validacao_RF.csv",row.names = FALSE)


#teste valendo
classificacao_randomForest <- predict(mod_rf,teste)

outputrandomForest <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_randomForest)

any(is.na(outputrandomForest)) #procurando NA em treino


write.csv(outputrandomForest,"outputrandomForest.csv",row.names = FALSE)







#==================================================
#Gradient Boosting
set.seed(123)



treino_novo80$is_listened <- as.character(treino_novo80$is_listened)
teste_novo20$is_listened <- as.character(teste_novo20$is_listened)

# gbm classe com formato character 2000 trees , 80
mod_gb <- gbm(is_listened ~ ., data = treino_novo80, n.trees = 2500, n.minobsinnode = 80,n.cores = 8, distribution = "bernoulli",verbose = TRUE)

classificacao_gb <- predict.gbm(mod_gb, teste_novo20[, -14], type = "response")
classificacao_gb <- as.data.frame(classificacao_gb)
classificacao_gb <- ifelse(classificacao_gb > 0.5, 1, 0)

matrizc_gradientb <- confusionMatrix(as.factor(classificacao_gb), as.factor(teste_novo20[,14]), positive = "1", mode = "prec_recall")
matrizc_gradientb$table



#metricas
precisao_gradientb <- matrizc_gradientb$byClass["Precision"]
recall_gradientb <- matrizc_gradientb$byClass["Recall"]
metricas_gradientb <- c(precisao_gradientb, recall_gradientb)
metricas_gradientb
metricas_gradientb <- data.frame(metricas_gradientb)

write.csv(metricas_gradientb, "Metricas_GradientBoost.csv",row.names = TRUE)


#escrevendo saida da validacao para juntar no ensamble 

saida_validacao_GB <- data.frame(classificacao_gb)
write.csv(saida_validacao_GB, "saida_validacao_GB.csv",row.names = FALSE)

#teste valendo
classificacao_gb <- predict.gbm(mod_gb,teste,type = "response")
output <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_gb)
write.csv(output, "outputGB.csv",row.names = FALSE)




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

#taxa de aprendizagem / eta = 0.4 depth = 10 rounds = 1500 
mod_xgboost <- xgboost(data = treino_novo80, label = aux_treino, max.depth = 40, eta = 0.4, nthread = 8, nrounds = 150, objective = "binary:logistic")
validacao <- predict(mod_xgboost,teste_novo20)

#ajustando e gerando matriz de confusão 
validacao <- ifelse(validacao > 0.5, 1, 0)
validacao <- sapply(validacao,as.factor)


tp1 <- sum((copia_saida_teste20 == 1) & (validacao == 1))
fp1 <- sum((copia_saida_teste20 == 0) & (validacao == 1))
tn1 <- sum((copia_saida_teste20 == 0) & (validacao == 0))
fn1 <- sum((copia_saida_teste20 == 1) & (validacao == 0))
matrizc_gxboost <- matrix(c(tn1, fn1, fp1, tp1), nrow = 2, ncol = 2, dimnames = list(c("0","1"), c("0","1")))

matrizc_gxboost


#metricas
#precisao XGBoost
precisao_XGBoost = tp1/(tp1+fp1)
precisao_XGBoost

#recall XGBoost
recall_XGBoost = tp1/(tp1+fn1)
recall_XGBoost

precisao_XGBoost <- as.data.frame(precisao_XGBoost)
recall_XGBoost <- as.data.frame(recall_XGBoost)


metricas_xgboost <- merge.data.frame(precisao_XGBoost,recall_XGBoost)

write.csv(metricas_xgboost, "Metricas_XGBoost.csv",row.names = TRUE)


#escrevendo saida da validacao para juntar no ensamble 

saida_validacao_XGB <- data.frame(validacao)
write.csv(saida_validacao_XGB, "saida_validacao_XGB.csv",row.names = FALSE)

testeXGBoost <- teste
sample_id <- testeXGBoost$sample_id
testeXGBoost$sample_id <- NULL
testeXGBoost <- as.matrix(testeXGBoost)


#teste valendo
classificacao_xgboost <- predict(mod_xgboost, testeXGBoost)

outputXGBoost <- data.frame(sample_id = teste$sample_id,is_listened = classificacao_xgboost)
write.csv(outputXGBoost, "outputXGBoost.csv",row.names = FALSE)


#==================================================
#media , media ponderada , voto

#gerando base com saidas do teste de cada algoritmo.

outputArvoreDsolo <- read.csv("outputArvoreDsolo.csv", stringsAsFactors = FALSE)
outputArvoreDsolo$sample_id <- NULL
outputGB <- read.csv("outputGB.csv", stringsAsFactors = FALSE)
outputGB$sample_id <- NULL
outputXGBoost <- read.csv("outputXGBoost.csv", stringsAsFactors = FALSE)
outputXGBoost$sample_id <- NULL

#media simples
media <- data.frame(sample_id = teste$sample_id,is_listened = ((outputArvoreDsolo$is_listened+outputGB$is_listened+outputXGBoost$is_listened)/3))
write.csv(media, "media.csv",row.names = FALSE)

#media ponderada peso = precisao medida
media_ponderada_precisao <- data.frame(sample_id = teste$sample_id,is_listened = ((outputArvoreDsolo$is_listened*0.689396761399741) + (outputGB$is_listened*0.716459773878542) +(outputXGBoost$is_listened*0.790259056656273)) /3 )
write.csv(media_ponderada_precisao, "media_ponderada_precisao.csv",row.names = FALSE)

#media ponderada peso = recall medido
media_ponderada_recall <- data.frame(sample_id = teste$sample_id,is_listened = ((outputArvoreDsolo$is_listened*1) + (outputGB$is_listened*0.945848960650944) +(outputXGBoost$is_listened*0.875637276714767)) /3 )
write.csv(media_ponderada_recall, "media_ponderada_recall.csv",row.names = FALSE)

#votacao
outputArvoreDsolo <- ifelse(outputArvoreDsolo > 0.5, 1, 0)
outputGB <- ifelse(outputGB > 0.5,1,0)
outputXGBoost <- ifelse(outputXGBoost > 0.5,1,0)
voto <- data.frame(sample_id = teste$sample_id,is_listened = outputArvoreDsolo+outputGB+outputXGBoost)
voto$is_listened <-ifelse(voto$is_listened >= 2,1,0 )

write.csv(voto, "voto.csv",row.names = FALSE)
