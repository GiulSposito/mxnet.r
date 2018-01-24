# setup
library(tidyverse)


### load database

# read function returns a list of datasets
load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <- load_image_file('./data/train-images.idx3-ubyte')
  test <- load_image_file('./data/t10k-images.idx3-ubyte')
  
  train$y <- load_label_file('./data/train-labels.idx1-ubyte')
  test$y <- load_label_file('./data/t10k-labels.idx1-ubyte')  
  
  return(
    list(
      train = train,
      test = test
    )
  )
}

# plot one case
show_digit <- function(arr784, col=gray(25:1/25), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# load
mnist <- load_mnist()

str(mnist)

# showing
labels <- paste(mnist$train$y[1:25],collapse = ", ")
par(mfrow=c(5,5), mar=c(0.1,0.1,0.1,0.1))
for(i in 1:25) show_digit(mnist$train$x[i,])

# show centroides
x_mean <- list()
for(i in 0:9) {
  x <- mnist$train$x[(mnist$train$y == i),]
  x_mean[[i+1]] <- colMeans(x) 
}

par(mfrow=c(2,5), mar=c(0.1,0.1,0.1,0.1))
lapply(X = x_mean, show_digit)

#
compare <- crossing(comp1 = 0:9, comp2 = 0:9)

res <- apply(compare, 1, function(x,m=x_mean){
  unlist(m[x[1]+1]) - unlist(m[x[2]+1])
}) %>% t() %>% as_tibble()

result <- bind_cols(compare, res)

par(mfrow=c(10,10), mar=c(0.1,0.1,0.1,0.1))
colFunc <- colorRampPalette(c("red","white","blue"))
res <- sapply(1:100, FUN=function(x) show_digit(as.matrix(result[x,3:786]),
                                                col=colFunc(35)))


dist <- apply(compare,1,function(x,m=x_mean){
  sqrt(mean((unlist(m[x[1]+1])-unlist(m[x[2]+1]))^2))  
}) %>% as_tibble()

res_dist <- bind_cols(compare,dist)

ggplot(res_dist, aes(x=comp1, y=comp2, fill=value)) +
  geom_tile() + 
  geom_text(aes(label=round(value))) +
  scale_fill_gradient2(low = "blue", high = "red") +
  scale_x_continuous(breaks=0:9) + 
  scale_y_continuous(breaks=0:9) + 
  theme_bw()
