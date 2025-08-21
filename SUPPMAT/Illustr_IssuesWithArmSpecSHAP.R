
###### illustrating that combining SHAP from individual models in T-learning tend to target the prognostic variables.
###### It is easy to see that inspecting (sh0_j + sh1_j) for each variable j is bad (where these are shaps from models 0 and 1)
###### But what about looking at differences (sh0_j - sh1_j)? See below that this not great either.
###### Strategy 3 in paper (surrogate SHAP) included too, see below.

require(lattice)
require(dplyr)
require(lattice)
require(latex2exp)
require(rlearner)
require(SHAPforxgboost)
source("functions.R")

set.seed(2025)
#################################

N.TREES <- 1000  ## both maxtrees for meta.learn,  and for SHAPs when using cvboost
EARLY.STOP = 50
PRINT.EVERY.N <- 50
VERB <- FALSE
DEBG <- FALSE 
DEBUG <- FALSE 
N.CV.FOLDS = 3 ## WAS 10 IN SHAP PAPER

NUMBER.OF.ITERATIONS <- 10

####### Toy example to highlight non-robustness against nuisance prognostic effects 
fun_simulate_data<- function(n.per.arm) { 
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
  lin <- with(DD,-1 + 3*(x1+x2+x3+x4+x5) + 1*as.numeric(trt)* 
                as.numeric(x7>0)*as.numeric(x8>0)  ##  x7 and x8 should be identified in SHAP rankings 
  )
  DD$y <- rnorm(n, lin, 0.5)  # it's just an example, 0.5 is arbitrary
  DD
}##EndFunction.

### Hardcoded for toy example: looks how well x7 and x8 did
fun.bestRanking <- function(dat, use="shap.diff") {
  dat$Y = dat[, which(names(dat)==use)]
  o <- order(abs(dat$Y), decreasing=TRUE)      
  tmp <- dat[o,]  
  where.x7 <- which(tmp$xvar== "x7") 
  where.x8 <- which(tmp$xvar== "x8") 
  where.best <- min(where.x7, where.x8)    
  where.best
}


#################################

SHAP.diffs.list <- vector("list", NUMBER.OF.ITERATIONS)
SHAP.surr.list <- vector("list", NUMBER.OF.ITERATIONS)
Metric.diffs.list <- vector("list", NUMBER.OF.ITERATIONS)
Metric.surr.list <- vector("list", NUMBER.OF.ITERATIONS)

for(iter in 1:NUMBER.OF.ITERATIONS) { 

  set.seed(iter)

  d <- fun_simulate_data(n.per.arm = 400)
  d.train.copy <- d

  (BIOMARKERS <- names(d)[grep(names(d), pattern="x", fixed=TRUE)])
  x <- as.matrix(d[,BIOMARKERS])
  w=d$trt 
  y <- d$y  
  k_folds_mu1=N.CV.FOLDS 
  k_folds_mu0=N.CV.FOLDS
  ntrees_max = N.TREES 
  print_every_n = PRINT.EVERY.N
  verbose=VERB 
  early_stopping_rounds=EARLY.STOP 

  # prepare for arm-spec outcome modelling, code from rlearner but here explicit to render SHAP too.
  input = rlearner:::sanitize_input(x, w, y)
  x = input$x
  w = input$w
  y = input$y
  x_1 = x[which(w == 1), ]
  x_0 = x[which(w == 0), ]
  y_1 = y[which(w == 1)]
  y_0 = y[which(w == 0)]
  nobs_1 = nrow(x_1)
  nobs_0 = nrow(x_0)
  pobs = ncol(x)
  
  t_1_fit = cvboost3(x_1, y_1, objective = "reg:squarederror", 
                   k_folds = k_folds_mu1, ntrees_max = ntrees_max,  
                   print_every_n = print_every_n, early_stopping_rounds = early_stopping_rounds, 
                   verbose = verbose, DEBUG=DEBUG)
  t_0_fit = cvboost3(x_0, y_0, objective = "reg:squarederror", 
                   k_folds = k_folds_mu0, ntrees_max = ntrees_max, 
                   print_every_n = print_every_n, early_stopping_rounds = early_stopping_rounds, 
                   verbose = verbose, DEBUG=DEBUG)
  sh0 <- shap.values(xgb_model = t_0_fit$xgb_fit, X_train = x) 
  sh1 <- shap.values(xgb_model = t_1_fit$xgb_fit, X_train = x) 

  ################################################
  ####### DIFFERENCES OF SHAP
  ################################################
  res2 <- data.frame(shap.diff = sh0$mean_shap_score- sh1$mean_shap_score,
                    xvar=c(names(sh0$mean_shap_score)))
  res2$xvar <- gsub(res2$xvar, pattern="covariate_", replace="x")
  res2$xvar <- factor(res2$xvar, levels=paste0("x", 1:20))
  SHAP.diffs.list[[iter]] <- res2
  res2$iter <- iter
  Metric.diffs.list[iter] <- fun.bestRanking(dat=res2, use="shap.diff")
  ################
  ## Surrogate SHAP would be derived like this: 
  x=X 
  w=d$trt 
  y <- d$y  
  ## Estimate CATE first
  tboost_fit <- tboost3(x=X, 
                        w=d$trt, 
                        y=d$y, 
                        k_folds_mu1=N.CV.FOLDS, 
                        k_folds_mu0=N.CV.FOLDS,
                        ntrees_max = N.TREES, 
                        print_every_n = PRINT.EVERY.N,
                        verbose=VERB, 
                        early_stopping_rounds=EARLY.STOP, 
                        DEBUG=DEBG);
  ## Now pretend CATE is just another variable and regress it against x:
  CATE.Tlearn = predict(tboost_fit, newx = X) 
  datFor2.T <- data.frame(cate=CATE.Tlearn, X) 
  ind.shap.model.T = cvboost3(x = X, 
                              y = datFor2.T$cate, 
                              objective = "reg:squarederror", 
                              weights = NULL, # no weights here! 
                              k_folds = N.CV.FOLDS, 
                              ntrees_max = N.TREES, 
                              print_every_n = 500, 
                              early_stopping_rounds = EARLY.STOP, 
                              verbose = VERB)
  ind.shaps.obj.T <- shap.values(xgb_model = ind.shap.model.T$xgb_fit, X_train = X) 
  sh.T <- ind.shaps.obj.T$mean_shap_score
  O <- order(sh.T, decreasing = FALSE)
  ssh.T <- sh.T[O]
  res3 <- data.frame(shap.surrogate = ssh.T,
                     xvar=names(ssh.T))
  res3$xvar <- factor(res3$xvar, levels=names(ssh.T))
  res3$iter <- iter
  SHAP.surr.list[[iter]] <- res3
  Metric.surr.list[iter] <-fun.bestRanking(dat=res3, use="shap.surrogate")
  
}##END.ITERATIONS

results_d <- do.call(rbind, SHAP.diffs.list); results_d <- as.data.frame(results_d)
head(results_d)           
results_s <- do.call(rbind, SHAP.surr.list); results_s <- as.data.frame(results_s)
head(results_s)           

rank_d <- do.call(rbind, Metric.diffs.list); 
rank_d <- as.numeric(rank_d)
#rank_d$what <- "ranking from SHAPDiffs"
head(rank_d)           
rank_s <- do.call(rbind, Metric.surr.list); 
rank_s <- as.numeric(rank_s)
#rank_s$what <- "ranking from surrSHAP"
head(rank_s)           

XL = c(0,20)

INFO = "Y= -1 + 3*(x1+x2+x3+x4+x5) + 1*as.numeric(trt)*
                sign(x7)*sign(x8), 20 x variables, n=400, T-learning"
INFO2 <- "\n Metric on both axes; metric is min(ranking(x7), ranking(x8)), 
  so lower is better (1=> x7 or x8 is top-ranked"
dev.off()
plot(rank_d, rank_s, xlim=XL, ylim=XL, main=paste("Simulation iterations from Model: \n",
                                                  INFO, INFO2), cex.main=0.7,
     xlab="DIFFERENCES OF ARM-SPEC SHAP",
     ylab="SURROGATE SHAP"); grid(); 
abline(0,1)

################################################################
trellis.par.set("superpose.symbol", list(pch=c(4,19)))
trellis.par.set("par.main.text", list(cex=0.8))

INFO <- "x1-x5 prog, x7,x8 pred"

dotplot(xvar ~ shap.diff, data=results_d, auto.key=list(space="right"),
        main=paste(INFO, "\n Diff  SHAP T-learning"))
dotplot(xvar ~ shap.surrogate, data=results_s, auto.key=list(space="right"),
        main=paste(INFO, "\n Surrogate SHAP T-learning"))





