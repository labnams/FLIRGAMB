# FLIRGAMB
Genetic algorithm code used for this article:Machine learning-based prediction of fatty liver diseases by gut microbial features in the presence of insulin resistance.

# Code description
- training set model.Ry : R code to build random forest model for classifying IRFL. This code is run with a toy example dataset
- GA_IRFL_k_10_toydata_github.py : Python code for feature selection to build optimal GA-IRFL model. This code is run with a toy example dataset.
- toydata.csv : toyset data to execute training model and genetic algorithm.

- test set model.Ry : R code to evaluate classification perfomance using the selected genera from GA-IRFL model. This code is run with a test dataset with the 10 selected genera.
- 10feature_rph_211021.csv : 10 selected genera data from the test dataset.

# Usage
- Run these python and R code with toydata.csv and 10feature_rph_211021.csv.

# Contact
- Mr. Baeki Kang (baekikang@gmail.com)
- Mr. Aron Park (parkar13@gmail.com)
- Prof. Seungyoon Nam (nams@gachon.ac.kr)
- Prof. Dongryeol Ryu (Dongryeol.Ryu@gmail.com)
