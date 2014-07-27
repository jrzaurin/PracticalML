library(caret)

training <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
#names of variables with ALL NAs
namesNA <- names(test)[seq_along(names(test))[sapply(test, function(x)all(is.na(x)))]]
#ignoring those variables for further analysis
subTraining <- training[,-which(names(training) %in% namesNA)]
subTraining <- subTraining[,-seq(1:7)]

#dividing training data into training and testing (or cross validation)
set.seed(1981)
inTrain = createDataPartition(subTraining$classe, p = 3/4)[[1]]
tTraining = subTraining[ inTrain,]
tCrossv = subTraining[-inTrain,]

#series of plots for exploratory analysis
pdf("AllPlots.pdf")
par(mfrow=c(2,2))
for (i in seq_len(ncol(tTraining))){
    plot(tTraining[,i], col = tTraining$classe)
}
dev.off()

#PCA analysis
svdTrain <- svd(scale(tTraining[,-c(ncol(tTraining))]))
PercVarExpl <- svdTrain$d^2/sum(svdTrain$d^2)


#simple function to capture a percentage of variable explained by PCs
fmaxPerc <- function(x, max=0.9) {
  s <- 0
  j <- 1
  for (i in seq_along(x)) {
      while (s <= max){
      s <- s + x[i]
      j <- j+1
      }
  }
  return(j-1)
}

#plot right singlular vectors
pdf("RSingVec.pdf")
par(mfrow=c(2,2))
for (i in seq_along(PercVarExpl)) {
    plot(svdTrain$v[,i],pch=19)
}
dev.off()

#Figure 1a and b in the report
par(mfrow=c(1,2))
plot(PercVarExpl,xlab = "Singular Value index",ylab = "Percentage of Variance",pch=19)
plot(svdTrain$v[,1],xlab = "Singular Value index",ylab = "1st Right singular Vector",pch=19)

#an alternative way of PCA analysis
prCompTrain <- prcomp(scale(tTraining[,-c(53)]))
summary(prComp)

#statistical analysis
library(nnet)
library(glmnet)
#method glmnet or multinom: performace issues. Extremely slow on a MacBook Pro 10.6.8
trainPC <- train(tTraining$classe ~ .,method="glmnet",preProcess="pca",data=tTraining)
confusionMatrix(tCrossv$classe,predict(trainPC,tCrossv))

#method rf: performace issues. Extremely slow on a MacBook Pro 10.6.8
trainRf <- train(tTraining$classe ~ .,method="rf",data=tTraining)
confusionMatrix(tCrossv$classe,predict(trainRf,tCrossv))

#method gbm: performace issues. Extremely slow on a MacBook Pro 10.6.8
trainGbm <- train(tTraining$classe ~ .,method="gbm",data=tTraining, verbose = FALSE)
confusionMatrix(tCrossv$classe,predict(trainGmb,tCrossv))


#method lda
trainLda <- train(tTraining$classe ~ .,method="lda",data=tTraining)
confusionMatrix(tCrossv$classe,predict(trainLda,tCrossv))

#Using a "simple" tree
library(tree)
trainTree <- tree(tTraining$classe ~ ., data = tTraining)
#to see the three classe: > trainTree
#for a plot: plot(trainTree); text(trainTree)
#in the plot below, upper axis is cost/complexity
plot(cv.tree(trainTree,FUN=prune.tree,method="misclass"))
#performance of a simple tree
predTest <- predict(trainTree, tCrossv, type = "class")
ConfusionMatrix(tCrossv$classe, predTest)


#random Forest
library(randomForest)
set.seed(921981)
#we set importance = TRUE for plotting importance later.
trainRf <- randomForest(tTraining$classe ~ . , data = tTraining, importance = TRUE)
#predict on a tCrossv test type = class and confusion Matrix
predCrossvRf <- predict(trainRf, tCrossv, type = "class")
confusionMatrix(tCrossv$classe,predCrossvRf)
#plot relative importance of variables
varImpPlot(trainRf)


#predict on the 20 obs test set
#note that given the nature of the modeling, the processing on
#test set is not neccesary since variables with all NAs will
#NOT be used
subTest <- test[,-which(names(test) %in% namesNA)]
subTest <- subTest[,-seq(1:7)]
predTestRf <- predict(trainRf, subTest, type = "class")














