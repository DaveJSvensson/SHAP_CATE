###################################################
###################################################
# Illustration of run times with KernalSHAP, and inspection
# against Surrogate SHAP.
###################################################
###################################################
rm(list=ls()) ## Use R 4.0.2
library(kernelshap)  # 0.7.0
library(randomForest) # 4.6-14
library(data.table) # 1.16.2 
library(xgboost) # 1.7.8.1
library(Matrix) # 1.2-18
require(SHAPforxgboost) # 0.1.3
#################################################
### GENERATE SOME DATA:
#################################################
fun.sim.baselineData <- function(N, p) {
  require(mvtnorm)
  COV <- diag(p)
  X <- rmvnorm(N,rep(0,p), COV)
  d <- data.frame(X)
  names(d) <- paste0("x", 1:p)
  d$trt <- rep(0:1, each=N/2)
  d
}#End
## -----------------------


set.seed(1)
n.rct <- 1000; SD=0.1
P <- 100 ## a bit more demanding for Kernel SHAP, larger values are slow. 

dd <- fun.sim.baselineData(N = n.rct, p=P)
dd$trt <- rep(c(0,1), each=n.rct/2); table(dd$trt);
ite <- 2*with(dd, as.numeric(x3>0)*as.numeric(dd$trt==1))
dd$y <- -1 + with(dd, x1+x2) + ite + rnorm(nrow(dd),0, SD)
head(dd,2)
    
dd.train <- dd

x_var <- paste0("x", 1:P)
x_var

y_var <- "y"
x_train <- dd[, x_var]
y_train <- dd[, y_var]

xy_train <- data.frame(x_train, y = y_train)
head(xy_train, 2)

  #################################################
  # ESTIMATE CATE : T-learning simplest possible.
  # This is just for illustration of SHAP strategies.
  # Normally, tuning/evaluations is considered for CATE 
  #################################################
    G.0 <- dd$trt == 0
    G.1 <- dd$trt == 1
    dd.0 <- xy_train[G.0,]
    dd.1 <- xy_train[G.1,]
    x_train.0 <- dd.0[, x_var]
    x_train.1 <- dd.1[, x_var]
    y.0 <- dd$y[G.0]
    y.1 <- dd$y[G.1]
    m.0 <- randomForest(y=y.0, x=x_train.0, ntree = 1000) ## keeping it super simple
    m.1 <- randomForest(y=y.1, x=x_train.1, ntree = 1000)
    dd$cate <- NA
    dd$cate <- predict(m.1, newdata = x_train) - predict(m.0, newdata = x_train)
    CATE = dd$cate
    datFor2 <- as.matrix(dd[,c("cate",x_var)])
    ################################
    
    m <- xgboost(data =datFor2[,-1], ## again no tuning. This is just for getting a fast result, illustrations!
             label=datFor2[,1],
             nrounds=100,
             nthread = 2
    )
    ## Summary SHAP:
    system.time(  
        IndirectSHAP <- shap.values(xgb_model = m, X_train = datFor2[,-1])
    )
  
    No.Sampling <- 2  ## Aes et al, sampling approximation
    st2 <- system.time(  # 22 s
        ks.2 <- kernelshap(m, datFor2[, x_var],  
                   bg_X = datFor2[sample(nrow(datFor2), No.Sampling),   
                                  x_var])#, exact = TRUE)
    ) 
    st2
    #user  system elapsed 
    #47.687   0.421  25.319 

    No.Sampling <- 10  ## Aes et al, sampling approximation
    st10 <- system.time(  # 22 s
      ks.10 <- kernelshap(m, datFor2[, x_var],  
                       bg_X = datFor2[sample(nrow(datFor2), No.Sampling),   
                                      x_var])#, exact = TRUE)
    ) 
    st10
    
    No.Sampling <- 200  ## Aes et al, sampling approximation
    st200 <- system.time(  # 22 s
      ks.200 <- kernelshap(m, datFor2[, x_var],  
                       bg_X = datFor2[sample(nrow(datFor2), No.Sampling),   
                                      x_var])#, exact = TRUE)
    ) 
    st200
    
    
    ################# INSPECT RESULTS BELOW, case with sampling=200
    
    KernelSHAP = list('shap_score','mean_shap_score','BIAS0')
    KernelSHAP$shap_score = ks.200$S
    KernelSHAP$mean_shap_score =  colMeans(abs(ks.200$S))

    SHAP.in <- IndirectSHAP$mean_shap_score
    SHAP.di <- KernelSHAP$mean_shap_score
      
    S1 <- data.frame(shap=SHAP.in, x=names(SHAP.in))
    S1$strategy <- "Surrogate SHAP"
    S2 <- data.frame(shap=SHAP.di, x=names(SHAP.di))
    S2$strategy <- "Kernal SHAP"

    RES <- rbind(S1,S2)
    RES$x <- factor(RES$x)
    RES$strategy <- factor(RES$strategy)
    require(lattice)
    dotplot(x ~ shap , group=strategy, data=RES) ## very similar global rankings
            
    ################ LOCAL SHAP:
    shap.in <- IndirectSHAP$shap_score
    shap.di <- KernelSHAP$shap_score

    sel.in <- data.frame(shap.in[, c("x2", "x3")]); 
    names(sel.in) <- paste0("sh.", names(sel.in)); sel.in$what <- "Kernal SHAP"
    sel.di <- data.frame(shap.di[, c("x2", "x3")]); 
    names(sel.di) <- paste0("sh.", names(sel.di)); 
    sel.di$what <- "Surrogate SHAP" 
    sel.x <- x_train[, c("x2", "x3")];

    sxx1 <- rbind(sel.in, sel.di)
    sxx2 <- rbind(sel.x,sel.x)
    res <- cbind(sxx1, sxx2)
    head(res,1)
    xyplot(sh.x3 ~ x3|what, data=res ) ## very similar
    xyplot(sh.x2 ~ x2|what, data=res ) ## very similar
    
    
    
