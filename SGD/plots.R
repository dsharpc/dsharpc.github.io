library(ggplot2)
rmse <- read.delim("RMSE_SGD.txt", sep = ",",header = TRUE)

ggplot(data=rmse) + geom_line(aes(x=iteration,y=rmse_t), col='blue') + geom_line(aes(x=iteration,y=rmse_v), col='red') + xlab("Iteration") + ylab("RMSE")
