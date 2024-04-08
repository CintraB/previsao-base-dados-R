# Projeto em R: Previsão de Listeners de Música

Este projeto em R tem como objetivo prever se os usuários irão ouvir uma determinada música com base em diferentes características. Para isso, foram utilizadas técnicas de aprendizado de máquina, incluindo árvores de decisão, random forest, gradient boosting e XGBoost.

## Instalação das Bibliotecas

Para executar este projeto, é necessário instalar as seguintes bibliotecas do R:

```R
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
```

##Como Executar
Após instalar as bibliotecas necessárias, siga os passos abaixo para executar o projeto:

Carregue as bibliotecas necessárias:

```
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
```

Execute o código do projeto, que inclui a leitura dos dados, pré-processamento, treinamento dos modelos e avaliação.
Os modelos treinados serão usados para fazer previsões nos dados de teste e as saídas serão salvas em arquivos CSV.
Você pode então avaliar o desempenho dos modelos e explorar os resultados.

##Arquivos do Projeto

`train.csv`: Conjunto de dados de treinamento original.
`test.csv`: Conjunto de dados de teste original.
`treino.csv`: Conjunto de dados de treinamento após pré-processamento.
`teste.csv`: Conjunto de dados de teste após pré-processamento.
`treino_novo80.csv`: Conjunto de dados de treinamento com 80% das linhas selecionadas aleatoriamente.
`teste_novo20.csv`: Conjunto de dados de teste com 20% das linhas selecionadas aleatoriamente.

##Arquivos de Saída

`outputArvoreDsolo.csv`: Saída das previsões do modelo de Árvore de Decisão para os dados de teste.
`outputrandomForest.csv`: Saída das previsões do modelo de Random Forest para os dados de teste.
`outputGB.csv`: Saída das previsões do modelo de Gradient Boosting para os dados de teste.
`outputXGBoost.csv`: Saída das previsões do modelo de XGBoost para os dados de teste.

##Avaliação dos Modelos

`Metricas_ArvoreD.csv`: Métricas de avaliação do modelo de Árvore de Decisão.
`Metricas_RandomForest.csv`: Métricas de avaliação do modelo de Random Forest.
`Metricas_GradientBoost.csv`: Métricas de avaliação do modelo de Gradient Boosting.
`Metricas_XGBoost.csv`: Métricas de avaliação do modelo de XGBoost.

##Ensambles

`media.csv`: Resultado do ensemble utilizando a média das previsões dos modelos.
`media_ponderada_precisao.csv`: Resultado do ensemble ponderado pela precisão das previsões dos modelos.
`media_ponderada_recall.csv`: Resultado do ensemble ponderado pelo recall das previsões dos modelos.
`voto.csv`: Resultado do ensemble por voto majoritário dos modelos.

##Melhorias Futuras

Para melhorar ainda mais a performance do modelo, algumas sugestões:

- Explorar mais técnicas de pré-processamento de dados.
- Testar diferentes parâmetros nos modelos de aprendizado de máquina.
- Experimentar com outras técnicas de ensemble, como boosting e bagging.
