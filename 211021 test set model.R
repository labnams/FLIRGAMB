#210701 test set model.R

#Loading library
Sys.getlocale()
Sys.setlocale("LC_ALL", "English_US")
library(pROC)
library(caret)
library(dplyr)
library(gbm)

#Initial setting
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = T,
  savePredictions = T,
  summaryFunction = twoClassSummary)
tunegr<-expand.grid(mtry=c(1:10))

#Dataset preparation and partition
dataset <- read.csv("GA_selected_10feature_rph_211021.csv", header = T)
inTrain <- createDataPartition( y = dataset$class_1, p = .80, list = FALSE)
training <- dataset [inTrain, ]
testing <- dataset [-inTrain,]

#RPM (rate per million) conversion for training dataset
training_numeric<-Filter(is.numeric, training)
training_factor<-Filter(is.character, training)

for (i in 1:nrow(training_numeric)){
  training_numeric[i,] <- 1000000*(training_numeric[i,]/sum(training_numeric[i,]))
}
training_2<-cbind(training_factor, training_numeric)
training_2 [is.na(training_2)] <-0

#RPM (rate per million) conversion for testing dataset
testing_numeric<-Filter(is.numeric, testing)
testing_factor<-Filter(is.character, testing)

for (i in 1:nrow(testing_numeric)){
  testing_numeric[i,] <- 1000000*(testing_numeric[i,]/sum(testing_numeric[i,]))
}
testing_2<-cbind(testing_factor, testing_numeric)
testing_2 [is.na(testing_2)] <-0


#ML prediction
model <- train(class_1 ~ ., data = training_2, method ="rf", preProcess=c("center","scale", "zv"), 
                    trControl = ctrl, metric = "ROC", tuneGrid=tunegr)

#Prediction using testing dataset
prediction<-predict(model, testing_2)
real_class <- as.factor(testing_2$class_1)
confusionMatrix(prediction, real_class)
