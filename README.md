## SHAP_CATE
This repository contains code for the tutorial paper 'Overview and practical recommendations on using Shapley Values for identifying predictive biomarkers via CATE modeling' (https://arxiv.org/abs/2505.01145).
Authors: David Svensson, Erik Hermansson, Nikolaos Nikolaou, Konstantinos Sechidis, Ilya Lipkovich
Contact: david.j.svensson@astrazeneca.com

## Overview
The paper investigates various methods for deriving SHAP values in the context of different CATE (Conditional Average Treatment Effect) modelling strategies, focusing on popular meta-learners including T-learning, S-learning, X-learning, R-learning, and DR-learning. It presents both illustrative examples and extensive simulation-based comparisons to evaluate these approaches. The accompanying GitHub code is intended primarily for demonstrating individual runs and workflows; however, the full-scale benchmarking described in the paper is not included in the repository, as it demands significant computational resources and parallel processing capabilities.

# Installation requirements/platform:
R version 4.0.2 (2020-06-22)
Platform: x86_64-pc-linux-gnu (64-bit)
Specific packages used: SHAPforxgboost_0.1.3 rlearner_1.1.0       latex2exp_0.9.6      dplyr_1.1.4          lattice_0.20-41

# Repository Structure
Contains code related to Section 3-5 in the manuscript, in particular to the different SHAP strategies for CATE modeling. 
FUNCTIONS: contains R functions used throughout the other scripts, and needs to be sourced before running.
PART1: Examples of SHAP strategy 1 (Kernel SHAP) and illustrations in Section 3.
PART2: Examples of SHAP strategy 2 and 3 (reducible and irreducible CATE models, with direct and indirect derivation of SHAP), related to Section 4 and 5.   
CASESTUDY: Code for real data applications.
SUPP.MAT: Some additional experiments related to the supplementary material 

 

