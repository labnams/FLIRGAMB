Sys.getlocale()
Sys.setlocale("LC_ALL", "English_US")
library(pROC)
library(caret)
library(dplyr)
library(gbm)
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = T,
  savePredictions = T,
  summaryFunction = twoClassSummary)
set.seed(2012140071)

dataset <- read.csv("toydata.csv", header = T)
#Patient id removal
dataset <- dataset %>% select(-patient)
#Unidentified bacteria removal
dataset <- dataset[,-ncol(dataset)]
#nearzerovar removal
dataset <- dataset[, -nearZeroVar(dataset)]

#RPM (rate per million) conversion for dataset
dataset_numeric<-Filter(is.numeric, dataset)
dataset_factor<-Filter(is.character, dataset)

for (i in 1:nrow(dataset_numeric)){
  dataset_numeric[i,] <- 1000000*(dataset_numeric[i,]/sum(dataset_numeric[i,]))
}
dataset_2<-cbind(dataset_factor, dataset_numeric)
dataset_2 [is.na(dataset_2)] <-0

#Dataset partition
inTrain <- createDataPartition( y = dataset_2$class_1, p = .80, list = FALSE)
training <- dataset_2 [inTrain, ]
testing <- dataset_2 [-inTrain,]

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

#Initial Model development
model_1 <- train(class_1 ~ ., data = training_2, method ="rf", preProcess=c("center","scale", "zv"), trControl = ctrl, metric = "ROC")
model_1_importance <- varImp(model_1, scale = FALSE)

#Initial feature selection and prepare dataset with selected feature (RPM normalized)
model_1_var1 <- as.data.frame(model_1_importance$importance)
model_1_var2 <- arrange(model_1_var1, desc(Overall))
model_1_var3 <- model_1_var2 %>% head(32)
model_1_var3[is.na(model_1_var3)] <- 0
model_1_var4 <- row.names(model_1_var3)
model_1_var5 <- training_2[,model_1_var4]
model_1_var4 <-as.data.frame(model_1_var4)
model_1_var4 <- cbind(model_1_var4,model_1_var3)

for (i in 1:nrow(model_1_var5)){
  model_1_var5[i,]<-as.numeric(model_1_var5[i,])
} 
for (i in 1:nrow(model_1_var5)){
  model_1_var5[i,]<-1000000*(model_1_var5[i,]/sum(model_1_var5[i,]))
}
model_1_var5[is.na(model_1_var5)] <- 0

training_2$class_1<-as.factor(training_2$class_1)
model_1_var6<-training_2$class_1
model_1_var5 <- cbind(model_1_var6,model_1_var5)

###########################################################
# Model development using selected 32 features 
tunegr<-expand.grid(mtry=c(4,8,12,16,20,24,28,32))
model_2 <- train(model_1_var6 ~ . , data = model_1_var5, method ="rf", preProcess=c("center","scale", "zv"), trControl = ctrl, metric = "ROC", tuneGrid=tunegr)
model_2_importance <- varImp(model_2, scale = FALSE)
#Secondary feature selection and prepare dataset with selected feature (RPM normalized)
model_2_var1 <- as.data.frame(model_2_importance$importance)
model_2_var2 <- arrange(model_2_var1, desc(Overall))
model_2_var3 <- model_2_var2 %>% head(28)
model_2_var3[is.na(model_2_var3)] <- 0
model_2_var4 <- row.names(model_2_var3)
model_2_var5 <- training_2[,model_2_var4]
model_2_var4 <-as.data.frame(model_2_var4)
model_2_var4 <- cbind(model_2_var4,model_2_var3)

for (i in 1:nrow(model_2_var5)){
  model_2_var5[i,]<-as.numeric(model_2_var5[i,])
} 
for (i in 1:nrow(model_2_var5)){
  model_2_var5[i,]<-1000000*(model_2_var5[i,]/sum(model_2_var5[i,]))
}
model_2_var5[is.na(model_2_var5)] <- 0

training_2$class_1<-as.factor(training_2$class_1)
model_2_var6<-training_2$class_1
model_2_var5 <- cbind(model_2_var6,model_2_var5)
###########################################################

# Model evaluation using test dataset

#Prediction using testing dataset
prediction<-predict(model_2, testing_2)
real_class <- as.factor(testing_2$class_1)
confusionMatrix(prediction, real_class)