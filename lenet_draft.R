library(mxnet)
library(magrittr)
library(caret)

source("./mnist/mnist_reader.R")

mnist <- load_mnist()

# pipe assign function
# example:  rnorm(5,mean=5) %>% sqrt() %=>% "varname" %>% mean()


"%=>%" <- function(val,var) {
  assign(substitute(var),val, envir = .GlobalEnv)
  return(val)
}


lenet <- mx.symbol.Variable("data") %>%
  # conv1
  mx.symbol.Convolution( kernel=c(5,5), num_filter=20, name="Conv1" ) %=>% "Conv1" %>%
  mx.symbol.Activation( act_type="tanh", name="Act1" ) %=>% "Act1" %>%
  mx.symbol.Pooling( pool_type="max",
                     kernel=c(2,2), stride=c(2,2),
                     name = "Pool1") %=>% "Pool1" %>%
  # conv2
  mx.symbol.Convolution( kernel=c(5,5), num_filter=50 , name="Conv2"  ) %=>% "Conv2" %>%
  mx.symbol.Activation( act_type="tanh", name="Act2" ) %=>% "Act2" %>%
  mx.symbol.Pooling( pool_type="max", 
                     kernel=c(2,2), stride=c(2,2),
                     name = "Pool2") %=>% "Pool2" %>%
  
  mx.symbol.flatten( name="Flat") %=>% "Flat1" %>%
  
  # Layer Full 1
  mx.symbol.FullyConnected( num_hidden=500, name="Full1" ) %=>% "Full1" %>%
  mx.symbol.Activation( act_type="tanh", name="Act3" ) %=>% "Act3" %>%
  
  # Layer Full 2
  mx.symbol.FullyConnected( num_hidden=10 , name="Full2") %=>% "Full2" %>%
  mx.symbol.SoftmaxOutput(name="SoftM") %=>% "SoftM" 


graph.viz( lenet )

tr.x <- t(mnist$train$x)
dim(tr.x) <- c(28,28,1,ncol(tr.x))
ts.x <- t(mnist$test$x)
dim(ts.x) <- c(28,28,1,ncol(ts.x))

logmx <- mx.callback.log.train.metric(100)
mx.set.seed(42)
ti <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=tr.x, y=mnist$train$y,
                                     ctx=mx.cpu(), num.round=2, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=logmx)
te <- proc.time()
print(te-ti)

mx.model.save(model, "./mnist/lenet", 3)

model <- mx.model.load("./mnist/lenet", 3)


out <- mx.symbol.Group(c(Conv1, Act1, Pool1, Conv2, Act2, Pool2, Flat1, Full1, Full2, SoftM))
executor <- mx.simple.bind(symbol = out,  data=dim(ts.x), ctx=mx.cpu())

lapply(model$arg.params,dim)
lapply(executor$arg.arrays,dim)


mx.exec.update.arg.arrays(executor, model$arg.params, match.name = T)
mx.exec.update.aux.arrays(executor, model$aux.params, match.name = T)

mx.exec.update.arg.arrays(executor, list(data=mx.nd.array(ts.x)), match.name=TRUE)

mx.exec.forward(executor, is.train=FALSE)
names(executor$ref.outputs)

j <- 4

par(mfrow=c(1,1), mar=c(0.1,0.1,0.1,0.1))
image(ts.x[,28:1,1,j])

# Plot the filters of the 7th test example
par(mfrow=c(4,5), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:20) {
  outputData <- as.array(executor$ref.outputs$Conv1_output)[,,i,j]
  image(outputData[,24:1],
        xaxt='n', yaxt='n')
}

# Plot the filters of the 7th test example
par(mfrow=c(4,5), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:20) {
  outputData <- as.array(executor$ref.outputs$Act1_output)[,,i,j]
  image(outputData[,24:1],
        xaxt='n', yaxt='n')
}


for (i in 1:20) {
  outputData <- as.array(executor$ref.outputs$Pool1_output)[,,i,j]
  image(outputData[,12:1],
        xaxt='n', yaxt='n')
}

# Plot the filters of the 7th test example
par(mfrow=c(7,7), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:49) {
  outputData <- as.array(executor$ref.outputs$Conv2_output)[,,i,j]
  image(outputData[,8:1],
        xaxt='n', yaxt='n')
}

# Plot the filters of the 7th test example
par(mfrow=c(7,7), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:49) {
  outputData <- as.array(executor$ref.outputs$Act2_output)[,,i,j]
  image(outputData[,8:1],
        xaxt='n', yaxt='n')
}

# Plot the filters of the 7th test example
par(mfrow=c(7,7), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:49) {
  outputData <- as.array(executor$ref.outputs$Pool2_output)[,,i,j]
  image(outputData[,4:1],
        xaxt='n', yaxt='n')
}


outputData <- as.array(executor$ref.outputs$SoftM_output)[,j]

par(mfrow=c(1,1), mar=c(0.1,0.1,0.1,0.1))
plot(1:10,outputData)





x <- mnist$train$x
x <- x[1,]
dim(x)<- c(28,28)
View(x)
View(tr.x[,,1,1])
