########################################################
## Code illustrating derivation of strategy 2 and 3 for DR-learning.
## (See Section 'Strategies for deriving SHAP explanations from CATE estimators' in the paper.)
## (Only one iteration here; the full run takes long time).
########################################################
# All packages and versions are stored in the renv file. 
# The environment can be loaded with the command renv::restore(), this will install all the packages with the right version

source("funcs.R") ## Support functions
require(rlearner) ## See Renv file for version
require(xgboost) ## See Renv file for version
require(rlearner) ## See Renv file for version
require(SHAPforxgboost) ## See Renv file for version
require(lattice) 

# Simulation settings to vary 
model="S2" # Simulation model used here for illustration (Both S2 and S3 was used in paper)
# GET THE SIM MODEL 
fun_simulate_data <-get_sim_model(model)

NN.RCT <- 1000 # Number of patients in the data set

### FIXED SETTING FOR ALL SIMS
NN.RCT.TESTDATA <- 10000 ## Sample size for test data (Fixed regardless of NN.RCT)
N.TREES <- 10000  ## 10000 was used in paper, for both maxtrees for meta.learn, and for SHAPs when using cvboost
EARLY.STOP = 50 # early stop for xgboost
PRINT.EVERY.N <- 500
VERB <- FALSE
DEBG <- FALSE 
N.CV.FOLDS = 3 ## WAS 10 IN SHAP PAPER

INIT.TIME <- date()

#for (BETA in BETAS){ ## we only show one of the BETAS in this script.

seed=1 ## careful with this, if parallel runs, then each iteration needs its unique value.

BETA <- 1  ## This corresponds to section 4.3, prognostic nuisance parameter.
  ## PREPARE DATA
  d <- fun_simulate_data(N=NN.RCT, BETA = BETA, seed = seed)
  d0 <- fun_simulate_data(N=NN.RCT.TESTDATA, BETA = BETA, seed = seed) ## testdata
  ## #######################
  d$y <- d$y.obs; d$y.obs <- NULL
  d0$y <- d0$y.obs; d0$y.obs <- NULL
  
  d <- fun.Categ2DummyCoding(data=d, x_cat_var = "x2")
  d0 <- fun.Categ2DummyCoding(data=d0, x_cat_var = "x2")
  
  d$x2 <- NULL 
  d0$x2 <- NULL 
  
  names(d)[names(d)=="ite"] <- "true.ite"
  names(d0)[names(d0)=="ite"] <- "true.ite"
  
  d.train.copy <- d ## using this sometimes, see below
  d.test.copy <- d0 
  
  ## Training and Test baseline biomarker data
  (BIOMARKERS <- names(d)[grep(names(d), pattern="x", fixed=TRUE)])
  X <- as.matrix(d[,BIOMARKERS])
  X0 <- as.matrix(d0[,BIOMARKERS])
  
  ###################################### CATE BELOW; AND LATER SHAP ######################
  # Set propensity scores depending on if it is RWD or not
  if (model == "S2"){
    # Propensity score for model
    pA <- mean(d.train.copy$trt)
    pA2 <- mean(d.test.copy$trt)
    pre.prop <- rep(pA, nrow(d)) ## vector, needed sometimes
    pre.prop0 <- rep(pA2, nrow(d0)) ## vector, needed sometimes 
    
    print(mean(pre.prop))
    
  } else if (model == "S3"){
    pre.prop <- NULL
    pre.prop0 <- NULL
  }else {
    stop("Model needs to be either S2 or S3")
  }
  
  ### ----------------------------------------------------------------------------
  ## The following is just a trick, required because direct and indirect results in different naming conventions:
  ## try it in the console to see why. 
  ####### i.e., cvboost is using  rlearner:::sanitize_input(X, d$trt, d$y) which renames x
  ###### so now we can see that e.g., "covariate_2" is actually "x3" (all this due to x2a, x2b).
  ## -----------------------------------------------------------------------------------
  
  input = rlearner:::sanitize_input(X, d$trt, d$y) ## don't need this object per se
  xx = input$x ## but this is needed: 
  dictionary <- data.frame(colnames(xx), colnames(X))

  ### ---------------------------
  drboost_fit.shaps <- DRboost3_shaps(x=X, 
                                      w=d$trt, 
                                      y=d$y, 
                                      p_hat=pre.prop,  ## pre needed (RCT)!
                                      k_folds = N.CV.FOLDS,
                                      ntrees_max = N.TREES, 
                                      print_every_n = PRINT.EVERY.N,
                                      verbose=VERB, 
                                      early_stopping_rounds=EARLY.STOP,
                                      DEBUG=DEBG);
  
  ## /////////// Predict Training data, used for indirect SHAPs later ///////////////////
  
  d.train.copy$CATE.DRlearn = drboost_fit.shaps$cate.est  
  datFor2.DR <- data.frame(cate=d.train.copy$CATE.DRlearn, X)  

  #########################
  
  ###_________________________
  sh.DR0 <- drboost_fit.shaps$dir.shaps.obj$mean_shap_score ### CHECK!!
  sh.DR1 <- drboost_fit.shaps$ind.shaps.obj$mean_shap_score ### CHECK!!
  
  names(sh.DR0) <- translate_what_covariate_is_what_x(ranking = sh.DR0, dictionary = dictionary)
  names(sh.DR1) <- translate_what_covariate_is_what_x(ranking = sh.DR1, dictionary = dictionary)
  
 #### Focus on metrics TOP1, NET3, Margin. Rankings can differ in the noisy variables. But top ones are similar. In the paper we ran many iterations.
 par(mfrow=c(2,1))
 barplot(sh.DR0, las=2)
 barplot(sh.DR1, las=2)
 
