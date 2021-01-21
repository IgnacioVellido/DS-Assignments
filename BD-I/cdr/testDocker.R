# Test libraries
library(tidyverse)
library(RSNNS)
library(frbs)
library(FSinR)
library(forecast)
library(caret)

# Test ggplot (and tidyverse)
ggplot(iris, aes(x=Sepal.Length, y=Petal.Length))
ggsave("testGgplot.png")

# Test caret
learn_model <-function(dataset, ctrl, message){
  model.fit <- caret::train(Class ~ ., data = dataset, method = "knn",
                   trControl = ctrl, preProcess = c("center","scale"), metric="ROC",
                   tuneGrid = expand.grid(k = c(1,3,5,7,9,11)))
  model.pred <- predict(model.fit,newdata = dataset)
  model.cm <- caret::confusionMatrix(model.pred, dataset$Class,positive = "positive")
  model.probs <- predict(model.fit,newdata = dataset, type="prob")

  return(model.fit)
}

df <- iris
df$Class <- ifelse(df$Species == "virginica", "positive", "negative") %>% as.factor()
df$Species <- NULL

ctrl <- trainControl(method="repeatedcv",number=5,repeats = 3,
                     classProbs=TRUE,summaryFunction = twoClassSummary)
model.raw <- learn_model(df, ctrl, "RAW ")

print(model.raw)