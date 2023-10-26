# Simulation to show bias & MSPE in variable selection and shrinkage

#- Generate data from lin reg model with 1 important var and 9 zeroes
#- Fit MLR and estimate all parameters --- show properties
#- Use stepwise to select best model
#- Estimate parameters only if chosen, otherwise record 0
#- show properties
#- Use LASSO to select and estimate
#- Show properties

library(stringr)
library(glmnet)
library(MASS)

set.seed(3014312)
n <- 50 #  PLAY WITH THIS NUMBER
p <- 10 #  PLAY WITH THIS NUMBER
beta1 <- 1
sigma <- 3 # PLAY WITH THIS NUMBER
iter <- 100

coefs.lm <- matrix(NA, nrow = iter, ncol = p + 1)
coefs.sw <- matrix(NA, nrow = iter, ncol = p + 1)
coefs.ri <- matrix(NA, nrow = iter, ncol = p + 1)
coefs.la <- matrix(NA, nrow = iter, ncol = p + 1)
MSPEs <- matrix(NA, nrow = iter, ncol = 4)
colnames(MSPEs) <- c("LM", "STEP", "RIDGE", "LASSO")

## First generate some test data under the true model, and test X values
testx1 <- rnorm(n = 1000)
testx <- cbind(testx1, matrix(0, ncol = p-1, nrow = length(testx1)))
## the true model is that only X1 matters, all other predictors not in
## model, so set them to 0

testx.all <- cbind(testx1, matrix(rnorm(n = (p-1) * length(testx1)), 
                                  ncol = (p-1), nrow = length(testx1)))

## if don't know that only X1 in the model, include all of them

colnames(testx) <- paste0("X", 1:p)
colnames(testx.all) <- colnames(testx)


testy <- testx1 * beta1
## the test y, which only depends on x1, all other beta = 0


## Then fit an example test set of size n
## Example Data Set
x <- matrix(rnorm(n = n * p), nrow = n)
eps <- rnorm(n, 0, sigma)
y <- beta1 * x[, 1] + eps


curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-8, 8),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste(
    "One data set with n=", n, "\n Population R-squared=",
    round(1 / (1 + sigma^2), 2)
  )
)
points(x = x[, 1], y = y, pch = "X", col = "blue")


## then for 100 iterations fit each of the models to a new training set
## and look at the estimated coefficients and the MSPE on the test data
for (i in 1:iter) {
  x <- matrix(rnorm(n = n * p), nrow = n)
  eps <- rnorm(n, 0, sigma)
  y <- beta1 * x[, 1] + eps
  xydat <- data.frame(y, x)
  mod.lm <- lm(y ~ ., data = xydat)
  coefs.lm[i, ] <- coef(mod.lm)

  step1 <- step(
    object = lm(y ~ 1, data = xydat), scope = list(upper = mod.lm), direction = "both",
    k = log(nrow(xydat)), trace = 0
  )
  coef.locs <- c(1, 1 + as.numeric(str_remove_all(names(coef(step1))[-1], "X")))
  coefs.sw[i, ] <- 0
  coefs.sw[i, coef.locs] <- coef(step1)

  ridgec <- lm.ridge(y ~ ., lambda = seq(0, 100, .05), data = xydat)
  coefs.ri[i, ] <- coef(ridgec)[which.min(ridgec$GCV), ]

  cv.lasso.1 <- cv.glmnet(y = y, x = x, family = "gaussian")
  coefs.la[i, ] <- coef(cv.lasso.1)[, 1]

  pred.lm <- predict(mod.lm, as.data.frame(testx.all))
  pred.sw <- predict(step1, as.data.frame(testx.all))
  pred.ri <- as.matrix(cbind(1, testx.all)) %*% coef(ridgec)[which.min(ridgec$GCV), ]
  pred.la <- predict(cv.lasso.1, testx.all)

  MSPEs[i, ] <- c(
    mean((testy - pred.lm)^2),
    mean((testy - pred.sw)^2),
    mean((testy - pred.ri)^2),
    mean((testy - pred.la)^2)
  )
}


boxplot(MSPEs[, 1:2],
  main = paste0(
    "Comparison of MSPEs\n R-squared=",
    round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
  ),
  names = c("lm", "step")
)

min.t <- apply(MSPEs[, 1:2], 1, min)

boxplot(MSPEs[, 1:2] / min.t,
  main = paste0(
    "Comparison of Relative MSPEs\n R-squared=",
    round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
  ),
  names = c("lm", "step")
)


boxplot(MSPEs[, 1:3],
  main = paste0(
    "Comparison of MSPEs\n R-squared=",
    round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
  ),
  names = c("lm", "step", "ridge")
)

min.t <- apply(MSPEs[, 1:3], 1, min)

boxplot(MSPEs[, 1:3] / min.t,
  main = paste0(
    "Comparison of Relative MSPEs\n R-squared=",
    round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
  ),
  names = c("lm", "step", "ridge")
)


boxplot(MSPEs,
  main = paste0(
    "Comparison of MSPEs\n R-squared=",
    round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
  ),
  names = c("lm", "step", "ridge", "LASSO")
)

min.t <- apply(MSPEs, 1, min)

boxplot(MSPEs / min.t,
  main = paste0(
    "Comparison of Relative MSPEs\n R-squared=",
    round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
  ),
  names = c("lm", "step", "ridge", "LASSO")
)


## this will create many plots all at once!
## can run this for j from 1 to 11, will create plots
## for each beta, from beta0 to beta10.
## here just show beta1,beta2
## beta2 same behaviour as all others which should be 0


MSE <- matrix(NA, nrow = p + 1, ncol = 4)
for (j in 2:3) {
  truec <- ifelse(j - 1 == 1, 1, 0)

  boxplot(cbind(coefs.lm[, j], coefs.sw[, j], coefs.ri[, j], coefs.la[, j]),
    main = paste0(
      "Comparison of coefs for variable ", j - 1,
      "\n R-squared=", round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
    ),
    names = c("lm", "step", "Ridge", "LASSO")
  )
  abline(h = truec, col = "red")
  points(x = 1, y = mean(coefs.lm[, j]), col = "blue")
  points(x = 2, y = mean(coefs.sw[, j]), col = "blue")
  points(x = 3, y = mean(coefs.ri[, j]), col = "blue")
  points(x = 4, y = mean(coefs.la[, j]), col = "blue")

  boxplot(cbind(coefs.lm[, j], coefs.sw[, j], coefs.ri[, j]),
    main = paste0(
      "Comparison of coefs for variable ", j - 1,
      "\n R-squared=", round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
    ),
    names = c("lm", "step", "Ridge")
  )
  abline(h = truec, col = "red")
  points(x = 1, y = mean(coefs.lm[, j]), col = "blue")
  points(x = 2, y = mean(coefs.sw[, j]), col = "blue")
  points(x = 3, y = mean(coefs.ri[, j]), col = "blue")

  MSE[j, ] <- round(c(
    mean((coefs.lm[, j] - truec)^2),
    mean((coefs.sw[, j] - truec)^2),
    mean((coefs.ri[, j] - truec)^2),
    mean((coefs.la[, j] - truec)^2)
  ), 3)

  boxplot(cbind(coefs.lm[, j], coefs.sw[, j]),
    main = paste0(
      "Comparison of coefs for variable ", j - 1,
      "\n R-squared=", round(1 / (1 + sigma^2), 2), ", n=", n, ", p=", p
    ),
    names = c("lm", "step")
  )
  abline(h = truec, col = "red")
  points(x = 1, y = mean(coefs.lm[, j]), col = "blue")
  points(x = 2, y = mean(coefs.sw[, j]), col = "blue")

  par(mfrow = c(1, 2))
  hist(coefs.lm[, j],
    breaks = seq(-2.25, 2.25, 0.5) + truec, ylim = c(0, 100),
    main = paste0("Histogram of beta-hat", j-1, " from LS")
  )
  abline(v = truec, col = "red")
  abline(v = mean(coefs.lm[, j]), col = "blue")
  hist(coefs.sw[, j],
    breaks = seq(-2.25, 2.25, 0.5) + truec, ylim = c(0, 100),
    main = paste0("Histogram of beta-hat", j-1, " from Step")
  )
  abline(v = truec, col = "red")
  abline(v = mean(coefs.sw[, j]), col = "blue")

 
  par(mfrow = c(1, 2))
  hist(coefs.ri[, j],
    breaks = seq(-2.25, 2.25, 0.5) + truec, ylim = c(0, 100),
    main = paste0("Histogram of beta-hat", j-1, " from Ridge")
  )
  abline(v = truec, col = "red")
  abline(v = mean(coefs.ri[, j]), col = "blue")
  hist(coefs.la[, j],
    breaks = seq(-2.25, 2.25, 0.5) + truec, ylim = c(0, 100),
    main = paste0("Histogram of beta-hat", j-1, " from LASSO")
  )
  abline(v = truec, col = "red")
  abline(v = mean(coefs.la[, j]), col = "blue")
  par(mfrow = c(1, 1))
}

## this function gives use the proportion 
## of values in a vector which are non zero
nonz <- function(x) {
  mean(x != 0)
}


round(rbind(
  apply(X = coefs.lm, MARGIN = 2, FUN = nonz),
  apply(X = coefs.sw, MARGIN = 2, FUN = nonz),
  apply(X = coefs.la, MARGIN = 2, FUN = nonz)
), 2)

## all models estimate the intercept to be non zero, because it
## is not shrunk in any of the models (check the formula for LASSO, Ridge)
## stepwise doesn't set extra variables to be zero as much as LASSO does
## but only small difference


mean(coefs.sw[coefs.sw[, 2] != 0, 2])
## stepwise overestimates beta1 for the simulations where it is chosen 
## in the model

############################################
# Plot the predicted values vs. X1
## If we knew that only X1 mattered we could do this, but
## not the case in reality

preds.ls <- coefs.lm %*% t(cbind(1, testx))
preds.sw <- coefs.sw %*% t(cbind(1, testx))
preds.ri <- coefs.ri %*% t(cbind(1, testx))
preds.la <- coefs.la %*% t(cbind(1, testx))

par(mfrow = c(1, 2))

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-7, 7),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste0("Predictions from LS when p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.ls[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.ls <- apply(preds.ls, 2, mean)
points(x = testx1, y = meanp.ls, cex = 0.3, pch = 20, col = "blue")

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-7, 7),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste0("Predictions from step() when p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.sw[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.sw <- apply(preds.sw, 2, mean)
points(x = testx1, y = meanp.sw, cex = 0.3, pch = 20, col = "blue")


par(mfrow = c(1, 2))

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-7, 7),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste0("Predictions from Ridge, p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.ri[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.ri <- apply(preds.ri, 2, mean)
points(x = testx1, y = meanp.ri, cex = 0.3, pch = 20, col = "blue")

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-7, 7),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste0("Predictions from LASSO, p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.la[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.la <- apply(preds.la, 2, mean)
points(x = testx1, y = meanp.la, cex = 0.3, pch = 20, col = "blue")

###############################################################

## using all the predictors rather than just the true one

preds.ls.all <- coefs.lm %*% t(cbind(1, testx.all))
preds.sw.all <- coefs.sw %*% t(cbind(1, testx.all))
preds.ri.all <- coefs.ri %*% t(cbind(1, testx.all))
preds.la.all <- coefs.la %*% t(cbind(1, testx.all))

par(mfrow = c(1, 2))

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-8, 8),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste("Predictions from LS, p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.ls.all[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.ls.all <- apply(preds.ls.all, 2, mean)
points(x = testx1, y = meanp.ls.all, cex = 0.3, pch = 20, col = "blue")

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-8, 8),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste("Predictions from step(), p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.sw.all[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.sw.all <- apply(preds.sw.all, 2, mean)
points(x = testx1, y = meanp.sw.all, cex = 0.3, pch = 20, col = "blue")


par(mfrow = c(1, 2))

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-8, 8),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste("Predictions from Ridge, p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.ri.all[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.ri.all <- apply(preds.ri.all, 2, mean)
points(x = testx1, y = meanp.ri.all, cex = 0.3, pch = 20, col = "blue")

# Plot this line
curve(
  expr = beta1 * x, from = -3, to = 3,
  ylim = c(-8, 8),
  col = "red", lwd = 3, xlab = "X", ylab = "Y",
  main = paste("Predictions from LASSO, p=", p, " n=", n)
)
for (i in 1:iter) {
  points(x = testx1, y = preds.la.all[i, ], cex = 0.2, col = "lightblue")
}
abline(a = 0, b = 1, col = "red")
meanp.la.all <- apply(preds.la.all, 2, mean)
points(x = testx1, y = meanp.la.all, cex = 0.3, pch = 20, col = "blue")


## repeat this increasing p to p=25, see what happens
