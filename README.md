# FLIRGAMB
Training model, Genetic algorithm, and test model code used for this article:Machine learning-based prediction of fatty liver diseases by gut microbial features in the presence of insulin resistance.

# Code description
- training set model.R : R code to build random forest model for classifying IRFL. This code is run with a toy example dataset
- toydata.csv : toyset data to execute training model and genetic algorithm.
- GA_IRFL_k_10_toydata_github.py : Python code for feature selection to build optimal GA-IRFL model. This code is run with a toy example dataset.
- toydata_IRFL.csv : toyset data to execute genetic algorithm for classifying IRFL.
- GA_ISFL_k_8_toydata_github.py : Python code for feature selection to build optimal GA-IRFL model. This code is run with a toy example dataset.
- toydata_ISFL.csv : toyset data to execute genetic algorithm for classifying ISFL.

- test set model.R : R code to evaluate classification perfomance using the selected genera from GA-IRFL model. This code is run with a test dataset with the 10 selected genera.
- 10feature_rph_211021_coded.csv : 10 selected genera data from the test dataset.

# Usage
- Run these python and R code with toydata.csv and 10feature_rph_211021.csv.

# Contact
- Mr. Baeki Kang (baekikang@gmail.com)
- Mr. Aron Park (parkar13@gmail.com)
- Prof. Seungyoon Nam (nams@gachon.ac.kr)
- Prof. Dongryeol Ryu (Dongryeol.Ryu@gmail.com)
