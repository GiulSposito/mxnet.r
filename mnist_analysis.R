# setup
library(tidyverse)
library(factoextra)
library(plotly)
library(class)


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

# plot one case digit (by 1d array of 784 pixels)
show_digit <- function(arr784, col=gray(25:1/25), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# load
mnist <- load_mnist()

# look the data format
str(mnist)

# showing some cases
labels <- paste(mnist$train$y[1:25],collapse = ", ")
par(mfrow=c(5,5), mar=c(0.1,0.1,0.1,0.1))
for(i in 1:25) show_digit(mnist$train$x[i,])

# calc digits centroids
centroids <- list()
for(i in 0:9) {
  x <- mnist$train$x[(mnist$train$y == i),]
  centroids[[i+1]] <- colMeans(x) 
}

# ploting the centroids
par(mfrow=c(2,5), mar=c(0.1,0.1,0.1,0.1))
lapply(X = centroids, show_digit)

# compare cases
compare <- crossing(comp1 = 0:9, comp2 = 0:9)

# calc features differences between the centroids
res <- apply(compare, 1, function(x,m=centroids){
  unlist(m[x[1]+1]) - unlist(m[x[2]+1])
}) %>% t() %>% as_tibble()

centroids_diff <- bind_cols(compare, res)

# plot them
par(mfrow=c(10,10), mar=c(0.1,0.1,0.1,0.1))
colFunc <- colorRampPalette(c("red","white","blue"))
res <- sapply(1:100, FUN=function(x) show_digit(as.matrix(centroids_diff[x,3:786]),
                                                col=colFunc(35)))

# calculating the distance between the centroids in 786 dimensions
dist <- apply(compare,1,function(x,m=centroids){
  sqrt(mean((unlist(m[x[1]+1])-unlist(m[x[2]+1]))^2))  
}) %>% as_tibble()
centroids_dist <- bind_cols(compare,dist)

# ploting the distances
ggplot(centroids_dist, aes(x=comp1, y=comp2, fill=value)) +
  geom_tile() + 
  geom_text(aes(label=round(value))) +
  scale_fill_gradient2(low = "blue", high = "red") +
  scale_x_continuous(breaks=0:9) + 
  scale_y_continuous(breaks=0:9) + 
  theme_bw()

# applying PCA

# transforming in a matrix
tr.x <- mnist$train$x
ts.x <- mnist$test$x
all.x <- rbind(tr.x,ts.x)

# calculating the PCA
pca <- prcomp(all.x, center = T, scale. = T)
pca <- readRDS("pca.rds")

# rebuilding features (transformed)
tr.x <- pca$x[(1:mnist$train$n),]
ts.x <- pca$x[(mnist$train$n+1):(mnist$train$n+mnist$test$n),]

# principals component
fviz_eig(pca)

# distribuicao da % de sdev por
dev.off()
hist((pca$sdev)/sum(pca$sdev), breaks = 100, col="Red")

data_frame(x=1:784, sdev.cum=cumsum((pca$sdev)/sum(pca$sdev))) %>%
  plot_ly(x=~x,y=~sdev.cum) %>%
  add_lines()

# lets see the distribution
pca.idx <- sample(1:mnist$train$n, 200)
cases <- tibble(
  label = as.factor(mnist$train$y[pca.idx]),
  x = tr.x[pca.idx,1],
  y = tr.x[pca.idx,2],
  z = tr.x[pca.idx,3]
)

# all cases
cases %>%
  plot_ly(x=~x, y=~y, color=~label) %>%  
  add_markers()

# cases more "distant"
cases %>%
  filter(label %in% c("0", "1", "4")) %>%
  plot_ly(x=~x, y=~y, color=~label) %>%  
  add_markers()

# 3D
plot_ly(cases, x=~x, y=~y, z=~z, color=~label) %>%  
  add_markers()

# knn cross validation
part.idx <- sample(1:mnist$train$n, round(mnist$train$n/2))

# cross validation parameters
k <- seq(2,14, 2)
n <- seq(5,80,10)

cross.params <- crossing(k=k, n=n)[1:5,]

result <- apply(X = cross.params,1, function(p, tr.idx = part.idx, x=tr.x, y=as.factor(mnist$train$y)){
  k_par <- as.integer(p[1])
  n_par <- as.integer(p[2])
  
  print(paste0("fitting: k=",k_par, " n=",n_par))

  y_hat <- knn(
    train = x[tr.idx,1:n_par], 
    test=x[-tr.idx,1:n_par], 
    cl=y[tr.idx],
    k=k_par
  )
  
  accuracy <- mean(y[-tr.idx]==y_hat)
  print(paste0("Accuracy: ", round(accuracy,4)))
  return(accuracy)
  
})

