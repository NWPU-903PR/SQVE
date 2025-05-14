df1 <-read.csv('~/snp_AD.csv',header = TRUE)
mutation_rate <- apply(df1, 2, function(x) sum(x == 1) /length(x))  
write.csv(mutation_rate, "~/mutation_rate_AD.csv")
hist(mutation_rate, breaks=20, main="Mutation Rate Distribution", xlab="Mutation Rate", ylab="Frequency")   


#library(MASS)
fit <- fitdistr(mutation_rate, "normal")
shapiro.test(mutation_rate) 
curve(dnorm(x, mean=fit$estimate[1], sd=fit$estimate[2]), add=TRUE, col="blue")

expected <- length(mutation_rate) * dnorm(mutation_rate, mean=fit$estimate[1], sd=fit$estimate[2])
observed <- hist(mutation_rate, breaks=100, plot=FALSE)$counts
expected <- expected[1:length(observed)]

expected_prob <- expected / sum(expected)
chisq.test(observed, p=expected_prob)   

num_samples <- 500   # subject=500
sample_size <- 1000  # SNP=1000
mean_p <-fit$estimate[1]    
sd_p <-fit$estimate[2]    

p_values <- rnorm(num_samples, mean=mean_p, sd=sd_p)
p_values <- pmax(pmin(p_values, 1), 0)
#print(p_values)

samples <- matrix(0, nrow=num_samples, ncol=sample_size)
for (i in 1:num_samples) {
  samples[i,] <- rbinom(sample_size, size=1, prob=p_values[i])
}

cat("Generated Bernoulli distribution values:\n")
#print(samples)
samples<- data.frame(p_values, samples)
write.csv(samples, file="~/AD_500samples_1000snp.csv", row.names=FALSE)




