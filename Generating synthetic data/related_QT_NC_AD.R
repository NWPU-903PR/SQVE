##generating beta
pdf_normal1 <- function(x, mean, sd) {
  dnorm(x, mean, sd)
}
pdf_bimodal <- function(x, mean1, sd1, mean2, sd2) {
  a1 <- 0.1
  a2 <- 0.1
  pdf_normal1(x, mean1, sd1) * a1 + pdf_normal1(x, mean2, sd2) * a2
}
mean1 = -0.3
sd1 = 0.1
mean2 = 0
sd2 = 0.02
curve(pdf_bimodal(x, mean1, sd1, mean2, sd2), from = -0.7, to = 0.4, lwd = 1, col = "blue", xlab = "x", ylab = "PDF", main = "Distributuion")
beta_Unrelated1 <- rnorm(875, mean2, sd2)
beta_Related <- rnorm(25, mean1, sd1)
beta_Unrelated2 <- rnorm(100, mean2, sd2)
beta_all <- c(beta_Unrelated1, beta_Related, beta_Unrelated2)
write.csv(beta_all, "~/Data1/result_18_53/ROI53_beta.csv", row.names=FALSE)

##generating b
b_df <-read.csv('~/b/NC_b_estimate.csv',header = TRUE)
fit <- fitdistr(b_df$estimate, "normal")
mean_b <-fit$estimate[1]
sd_b <-fit$estimate[2]
hist(b_df$estimate,20)
curve(dnorm(x, mean=fit$estimate[1], sd=fit$estimate[2]), from = -3, to =9, col="blue", xlab = "x", ylab = "b_density", main = "b_Distributuion")
b <- rnorm(1, mean = mean_b, sd = sd_b)
write.csv(b, "~/Data1/result_18_53/ROI53_b.csv", row.names=FALSE)

#related_NC
data <- list()
for (i in 1:500) {
  filename <- paste("~/sim_data/mu/NC/NC", i, ".csv", sep="")
  data[[i]] <- read.csv(filename, header=TRUE)
}
mu_vec <- vector(length=500)
y_new_vec <- vector(length=500)
sample_val_vec <- vector(length=500)

for (i in 1:500) {
  x_matrix <- as.numeric(as.matrix(data[[i]]))
  mu <-  exp(beta_all %*% x_matrix + b)
  
  #sigma2
  roi_i<- read.csv('~/meansd_NC_roi.csv',header = TRUE)
  fit <- lm(sd ~ poly(mean, 2), data=roi_i)
  ggplot(data=roi_i, aes(x=mean, y=sd)) + 
    geom_point() +
    geom_smooth(method="lm", formula=y~poly(x,2), se=FALSE, color="red") +
    labs(title="Scatter Plot with Quadratic Regression", x="Mean", y="Standard Deviation (sd)")

  mean_val <- mu
  y_new <- predict(fit, newdata=data.frame(mean=mean_val))
  y_new <- abs(y_new)
  y_new <- round(y_new, 4)
  
  mean_val <- mu
  sd_val <- y_new
  fit_norm <- list(mean = mean_val, sd = sd_val)
  sample_val <- rnorm(n = 1, mean = fit_norm$mean, sd = fit_norm$sd)
  sample_val <- round(sample_val, 4)
  
  mu_vec[i] <- mu
  y_new_vec[i] <- y_new
  sample_val_vec[i] <- sample_val
}

output_df <- data.frame(mu=mu_vec, y_new=y_new_vec, sample_val=sample_val_vec)
write.csv(output_df, file="~/Data1/result_18_53/ROI53_NC_output.csv", row.names=FALSE)


#related_AD
data <- list()
for (i in 1:500) {
  filename <- paste("~/mu/AD/AD", i, ".csv", sep="")
  data[[i]] <- read.csv(filename, header=TRUE)
}
mu_vec <- vector(length=500)
y_new_vec <- vector(length=500)
sample_val_vec <- vector(length=500)

for (i in 1:500) {
  x_matrix <- as.numeric(as.matrix(data[[i]]))
  mu <-  exp(beta_all %*% x_matrix + b)
  
  #sigma2
  roi_i<- read.csv('~/sim_data/meansd_NC_roi.csv',header = TRUE)
  fit <- lm(sd ~ poly(mean, 2), data=roi_i)

  mean_val <- mu
  y_new <- predict(fit, newdata=data.frame(mean=mean_val))
  y_new <- abs(y_new)
  y_new <- round(y_new, 4)
  
  mean_val <- mu
  sd_val <- y_new
  fit_norm <- list(mean = mean_val, sd = sd_val)
  sample_val <- rnorm(n = 1, mean = fit_norm$mean, sd = fit_norm$sd)
  sample_val <- round(sample_val, 4)
  
  mu_vec[i] <- mu
  y_new_vec[i] <- y_new
  sample_val_vec[i] <- sample_val
}

output_df <- data.frame(mu=mu_vec, y_new=y_new_vec, sample_val=sample_val_vec)
write.csv(output_df, file="~/Data1/result_18_53/ROI53_AD_output.csv", row.names=FALSE)

















###单个样本输出
#x
x <-read.csv('~/sim_data/mu/CN/CN334.csv',header = TRUE)#row.names = 1
x_matrix <- as.numeric(as.matrix(x))
#class(x_matrix)  #"numeric"


#mu
mu <-  exp(beta_Unrelated %*% x_matrix + b)
#1/(1+exp(-x))


##sigma2
roi_i<- read.csv('~/sim_data/meansd_g1_roi.csv',header = TRUE)

# 进行二次多项式拟合
fit <- lm(sd ~ poly(mean, 2), data=roi_i)

# 绘制散点图和拟合曲线
ggplot(data=roi_i, aes(x=mean, y=sd)) + 
  geom_point() +
  geom_smooth(method="lm", formula=y~poly(x,2), se=FALSE, color="red") +
  labs(title="Scatter Plot with Quadratic Regression", x="Mean", y="Standard Deviation (sd)")


# 给定mu
x_new <-  mu

# 计算拟合y值
#y_new <- x_new + x_new^2
y_new <- abs(predict(fit, newdata=data.frame(mean=x_new)))
y_new <- round(y_new, 4)

# 打印结果
print(paste("The predicted sd value for mean = ", x_new, " is ", round(y_new, 4)))



#Y_roi
mean_val <- x_new
sd_val <- y_new

# 拟合正态分布
fit_norm <- list(mean = mean_val, sd = sd_val)

# 抽取一个值
sample_val <- rnorm(n = 1, mean = fit_norm$mean, sd = fit_norm$sd)
sample_val<- round(sample_val, 4)

# 打印结果
print(paste("The sample value from the normal distribution with mean =", mean_val, "and sd =", sd_val, "is", round(sample_val, 4)))


