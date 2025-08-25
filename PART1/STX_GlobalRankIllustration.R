###
rm(list=ls())
require(lattice)
require(dplyr)
require(lattice)
require(latex2exp)
require(rlearner)

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
                as.numeric(x7>0)*as.numeric(x8>0)
  )
  DD$y <- rnorm(n, lin, 0.5) 
  DD
}##EndFunction.

panel.superpose.lmline <- function (x, y, linecol = "black", 
                                    lwd = 2, col.grid = "grey80", ...) {
  panel.grid(-1, -1, col = col.grid)
  panel.superpose(x, y, ...)
  try(panel.lmline(x, y, col = linecol, lwd = lwd, ...))
}#EndFun


#################################
#################################
#################################
#################################
set.seed(123456)

d <- fun_simulate_data(n.per.arm = 400)
d.train.copy <- d

## Using observed proportions as propensity (when needed)
pA <- mean(d$trt)
pre.prop <- rep(pA, nrow(d)) ## vector, needed sometimes
X.INIT=date()

(BIOMARKERS <- names(d)[grep(names(d), pattern="x", fixed=TRUE)])
X <- as.matrix(d[,BIOMARKERS])
head(X,1)

#################################
#################################
#################################

#N.TREES <- 100  ## both maxtrees for meta.learn,  and for SHAPs when using cvboost
N.TREES <- 10000  ## both maxtrees for meta.learn,  and for SHAPs when using cvboost
EARLY.STOP = 10
PRINT.EVERY.N <- 5
VERB <- FALSE
DEBG <- FALSE 
N.CV.FOLDS = 3 ## WAS 10 IN SHAP PAPER

###############################
X.INIT=date()
xboost_fit = xboost3(x=X, 
                     w=d$trt, 
                     y=d$y, 
                     p_hat=pre.prop, ## pre needed (RCT)!
                     k_folds_mu0 = N.CV.FOLDS, 
                     k_folds_mu1=N.CV.FOLDS,
                     ntrees_max = N.TREES, 
                     print_every_n = 500,
                     verbose=VERB, 
                     early_stopping_rounds=EARLY.STOP, 
                     DEBUG=DEBG); 
### ---------------------------
sboost_fit = sboost3(x=X,
                     w=d$trt,
                     y=d$y,
                     k_folds = N.CV.FOLDS,
                     ntrees_max = N.TREES, 
                     print_every_n = PRINT.EVERY.N,
                     verbose=VERB, 
                     early_stopping_rounds=EARLY.STOP, 
                     DEBUG=DEBG); 

### ---------------------------
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
################################################

d.train.copy$CATE.Slearn = predict(sboost_fit, newx = X) #newx alt 
d.train.copy$CATE.Tlearn = predict(tboost_fit, newx = X) 
d.train.copy$CATE.Xlearn = predict(xboost_fit, newx = X, new_p_hat =pre.prop)  ## EH : need to be new_p_hat 
############## STACK DATA FOR VARIOUS USAGE LATER, INCLUDING SAVING TO DISC FOR POST-ANALYSIS

datFor2.S <- data.frame(cate=d.train.copy$CATE.Slearn, X) 
datFor2.T <- data.frame(cate=d.train.copy$CATE.Tlearn, X) 
datFor2.X <- data.frame(cate=d.train.copy$CATE.Xlearn, X) 

######################################################

###_________________________
ind.shap.model.S = cvboost3(x = X, 
                            y = datFor2.S$cate, 
                            objective = "reg:squarederror", 
                            weights = NULL, # no weights here! 
                            k_folds = N.CV.FOLDS, 
                            ntrees_max = N.TREES, 
                            print_every_n = 500, 
                            early_stopping_rounds = EARLY.STOP, 
                            verbose = VERB)
ind.shaps.obj.S <- shap.values(xgb_model = ind.shap.model.S$xgb_fit, X_train = X) 
sh.S <- ind.shaps.obj.S$mean_shap_score
#sh.S


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


ind.shap.model.X = cvboost3(x = X, 
                            y = datFor2.X$cate, 
                            objective = "reg:squarederror", 
                            weights = NULL, # no weights here! 
                            k_folds = N.CV.FOLDS, 
                            ntrees_max = N.TREES, 
                            print_every_n = 500, 
                            early_stopping_rounds = EARLY.STOP, 
                            verbose = VERB)
ind.shaps.obj.X <- shap.values(xgb_model = ind.shap.model.X$xgb_fit, X_train = X) 
sh.X <- ind.shaps.obj.X$mean_shap_score 
########################################
length(sh.T)
length(sh.S)
length(sh.X)

res <- data.frame(shap = c(sh.T, sh.S, sh.X),
                  x = c(names(sh.T),
                        names(sh.S), 
                        names(sh.X)),
                        what=rep(c("T", "S", "X"), each=20))
head(res,3)
res$x <- factor(res$x, levels=paste0("x", 1:20))


# Similar to Figure in the paper where it is discussed how to compare rankings across CATE models.
# E.g., the SHAP values per see (for a fixed x) should not numerically be compared across CATE model, it 
# is rather the inherent ranking within model that matters. In this example, S top-ranks predictive x, while T does not.
# Margins can also be considered (see Metric in paper), but first standardization within model is needed.
trellis.par.set("par.xlab.text", list(cex=1.1))
trellis.par.set("par.ylab.text", list(cex=1.1))
dotplot(x ~ shap, groups=what, data=res, auto.key=list(space="right"),
        scales=list(x=list(at=seq(0, 0.30, by=0.05), rot=0)),
        xlab=TeX(r'($\Phi_j$)'), ylab=TeX(r'($x_j$)'))




