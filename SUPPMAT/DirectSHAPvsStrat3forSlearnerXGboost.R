########## Illustrating DirectSHAP for an S-learner with XGboost as baselearner.
########## Also, SurrogateSHAP is derived.

require(lattice)
require(dplyr)
require(lattice)
require(latex2exp)
require(rlearner)
require(SHAPforxgboost)
source("functions.R")

N.TREES <- 1000  ## both maxtrees for meta.learn, and for SHAPs when using cvboost
EARLY.STOP = 100
PRINT.EVERY.N <- 50
VERB <- FALSE
DEBG <- FALSE 
N.CV.FOLDS = 5 ## WAS 10 IN SHAP PAPER

## ########################################################
## S-learning with boosting as baselearner (using cvboost3), 
## but directly derives SHAP internally. Modifies code from rlearner package.
sboost3.dirShap <- function (x, w, y, k_folds = NULL, ntrees_max = 1000, 
                             print_every_n = 100, early_stopping_rounds = 10,  
                             verbose = FALSE, DEBUG=FALSE) 
{
  require(rlearner)
  xx <- x
  input = rlearner:::sanitize_input(x, w, y)
  x = input$x
  w = input$w
  y = input$y
  nobs = nrow(x)
  pobs = ncol(x)
  if (is.null(k_folds)) {
    k_folds = floor(max(3, min(10, nobs/4)))
  }
  extData <- cbind(x, (w - 0.5) * x, (w - 0.5))
  colnames(extData) <- c(colnames(xx),
        paste0("trt",colnames(xx)), "trt"
  )
  s_fit = cvboost3(extData, y, objective = "reg:squarederror", 
                   k_folds = k_folds, ntrees_max = ntrees_max, 
                   print_every_n = print_every_n, early_stopping_rounds = early_stopping_rounds, 
                   verbose = verbose, DEBUG=DEBUG)
  dir.shaps.obj <- shap.values(xgb_model = s_fit$xgb_fit, X_train = extData) 
  extData0 <- cbind(x, (0 - 0.5) * x, (0 - 0.5))
  colnames(extData0) <- c( colnames(xx), paste0("trt",colnames(xx)),"trt"
  )
  extData1 <- cbind(x, (1 - 0.5) * x, (1 - 0.5))
  colnames(extData1) <- c( colnames(xx), paste0("trt",colnames(xx)),"trt"
  )
  
  mu0_hat = predict(s_fit, newx = extData0)
  mu1_hat = predict(s_fit, newx = extData1)
  tau_hat = mu1_hat - mu0_hat
  ###### Surrogate SHAP
  ind.shap.model.S = cvboost3(x = xx, 
                              y = tau_hat, 
                              objective = "reg:squarederror", 
                              weights = NULL, # no weights here! 
                              k_folds = N.CV.FOLDS, 
                              ntrees_max = N.TREES, 
                              print_every_n = 500, 
                              early_stopping_rounds = EARLY.STOP, 
                              verbose = VERB)
  ind.shaps.obj.S <- shap.values(xgb_model = ind.shap.model.S$xgb_fit, X_train = xx) 
  ret = list(s_fit = s_fit, mu0_hat = mu0_hat, mu1_hat = mu1_hat, 
             tau_hat = tau_hat, 
             directSHAP =dir.shaps.obj, 
             indirectSHAP = ind.shaps.obj.S
  )
  class(ret) <- "sboost"
  ret
}#Endfunction

## Simple simulation for illustration.
fun_simulate_data<- function(n.per.arm, Beta.main=3) { 
  n <- n.per.arm*2
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  x3 <- rnorm(n)
  x4 <- rnorm(n)
  x5 <- rnorm(n)
  x6 <- rnorm(n)
  x7 <- rnorm(n)
  x8 <- rnorm(n)
  x9 <- rnorm(n)
  x10 <- rnorm(n)
  x11 <- rnorm(n)
  x12 <- rnorm(n)
  x13 <- rnorm(n)
  x14 <- rnorm(n)
  x15 <- rnorm(n)
  x16 <- rnorm(n)
  x17 <- rnorm(n)
  x18 <- rnorm(n)
  x19 <- rnorm(n)
  x20 <- rnorm(n)
  trt <- rep(0:1, each=n/2)
  DD <- data.frame(trt, 
                   x1,x2,x3,x4,x5,
                   x6,x7,x8,x9,x10,
                   x11,x12,x13,x14,x15,
                   x16,x17,x18,x19,x20
  ) 
  DD$trt <- trt
  lin <- with(DD,-1 + Beta.main*(x1+x2+x3+x4+x5) + 1*as.numeric(trt)*
                as.numeric(x7>0)*as.numeric(x8>0)
  )
  DD$y <- rnorm(n, lin, 0.5) 
  DD
}##EndFunction.

#################################
#################################
#################################
#################################
set.seed(123456)

d <- fun_simulate_data(n.per.arm = 400)
(BIOMARKERS <- names(d)[grep(names(d), pattern="x", fixed=TRUE)])
X <- as.matrix(d[,BIOMARKERS])
head(X,1)

###############################
sboost_fit = sboost3.dirShap(x=X,
                     w=d$trt,
                     y=d$y,
                     k_folds = N.CV.FOLDS,
                     ntrees_max = N.TREES, 
                     print_every_n = PRINT.EVERY.N,
                     verbose=VERB, 
                     early_stopping_rounds=EARLY.STOP, 
                     DEBUG=DEBG); 

### ---------------------------
sh.S <- sboost_fit$indirectSHAP$mean_shap_score
sh.S
########################

par(mfrow=c(2,1))
barplot(sboost_fit$directSHAP$mean_shap_score, las=2,
        main="S-learn, SHAP directly derived from the CATE model,
        SimModel: TRUE.CATE=SIGN(x7)*SIGN(x8), strong main effects (x1,...,x5)", 
        cex.main=0.7)
barplot(sh.S, las=2, main="Surrogate SHAP (Strat 3)
                SimModel: TRUE.CATE=SIGN(x7)*SIGN(x8), strong main effects (x1,...,x5)", cex.main=0.7)



################## ANOTHER RUN BUT WITH WEAK MAIN EFFECTS:
set.seed(123456)
d <- fun_simulate_data(n.per.arm = 400, Beta.main = 0.1)
(BIOMARKERS <- names(d)[grep(names(d), pattern="x", fixed=TRUE)])
X <- as.matrix(d[,BIOMARKERS])
head(X,1)

sboost_fit2 = sboost3.dirShap(x=X,
                             w=d$trt,
                             y=d$y,
                             k_folds = N.CV.FOLDS,
                             ntrees_max = N.TREES, 
                             print_every_n = PRINT.EVERY.N,
                             verbose=VERB, 
                             early_stopping_rounds=EARLY.STOP, 
                             DEBUG=DEBG); 

### ---------------------------
sh.S2 <- sboost_fit2$indirectSHAP$mean_shap_score
sh.S2

########################

par(mfrow=c(2,1))
barplot(sboost_fit2$directSHAP$mean_shap_score, las=2,
        main="S-learn, SHAP directly derived from the CATE model,
        SimModel: TRUE.CATE=SIGN(x7)*SIGN(x8), weak main effects (x1,...,x5)", 
        cex.main=0.7)
barplot(sh.S2, las=2, main="Surrogate SHAP (Strat 3)
                SimModel: TRUE.CATE=SIGN(x7)*SIGN(x8), weak main effects (x1,...,x5)", cex.main=0.7)





