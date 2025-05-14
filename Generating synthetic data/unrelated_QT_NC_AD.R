num_iter <- 119
for (iter in 54:num_iter) {
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
  
  #beta_Related <- rnorm(100, mean1, sd1)
  beta_Unrelated <- rnorm(1000, mean2, sd2)
  #beta_all <- c(beta_Related, beta_Unrelated)
  beta_to_save <- data.frame(beta = beta_Unrelated)
  filename <- paste("~/Data1/result/beta_values_", iter, ".csv", sep = "")
  write.csv(beta_to_save, file = filename, row.names = FALSE)
  
  #generating b
  b_df <-read.csv('~/b/NC_b_estimate.csv',header = TRUE)
  fit <- fitdistr(b_df$estimate, "normal")
  mean_b <-fit$estimate[1]
  sd_b <-fit$estimate[2]
  b <- rnorm(1, mean = mean_b, sd = sd_b)
  b_to_save <- data.frame(b = b)
  filename <- paste("~/Data1/result/b_values_", iter, ".csv", sep = "")
  write.csv(b_to_save, file = filename, row.names = FALSE)
  
  #unrelated NC
  data_cn <- list()
  for (i in 1:500) {
    filename <- paste("~/mu/NC/NC", i, ".csv", sep="")
    data_cn[[i]] <- read.csv(filename, header=TRUE)
  }
  mu_vec_cn <- vector(length=500)
  y_new_vec_cn <- vector(length=500)
  sample_val_vec_cn <- vector(length=500)
  for (i in 1:500) {
    x_matrix <- as.numeric(as.matrix(data_cn[[i]]))
    mu <- exp(beta_Unrelated %*% x_matrix + b)
    #sigma2
    roi_i<- read.csv('~/meansd_g1_roi.csv',header = TRUE)
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
    
    mu_vec_cn[i] <- mu
    y_new_vec_cn[i] <- y_new
    sample_val_vec_cn[i] <- sample_val
  }
  output_df_cn <- data.frame(mu=mu_vec_cn, y_new=y_new_vec_cn, sample_val=sample_val_vec_cn)
  filename_cn <- paste("~/Data1/result/ROI", iter, "_CN_output.csv", sep="")
  write.csv(output_df_cn, file=filename_cn, row.names=FALSE)
  
  
  #unrelated AD
  data_ad <- list()
  for (i in 1:500) {
    filename <- paste("~/mu/AD/AD", i, ".csv", sep="")
    data_ad[[i]] <- read.csv(filename, header=TRUE)
  }
  
  # 计算AD:mu、y_new和sample_val值
  mu_vec_ad <- vector(length=500)
  y_new_vec_ad <- vector(length=500)
  sample_val_vec_ad <- vector(length=500)
  for (i in 1:500) {
    x_matrix <- as.numeric(as.matrix(data_ad[[i]]))
    mu <- exp(beta_Unrelated %*% x_matrix + b)
    
    #sigma2
    roi_i<- read.csv('~/meansd_g1_roi.csv',header = TRUE)
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
  
    mu_vec_ad[i] <- mu    
    y_new_vec_ad[i] <- y_new
    sample_val_vec_ad[i] <- sample_val
  }

  output_df_ad <- data.frame(mu=mu_vec_ad, y_new=y_new_vec_ad, sample_val=sample_val_vec_ad)
  filename_ad <- paste("~/Data1/result/ROI", iter, "_AD_output.csv", sep="")
  write.csv(output_df_ad, file=filename_ad, row.names=FALSE)
}
