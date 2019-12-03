# Health-Prediction

# About 

In today’s world, the datasets are so large and complex that traditional data processing application software is inadequate to deal with them. Challenges include capture, storage, analysis and data curation, search, sharing, transfer, visualization, querying, updating and information privacy. These types of data are often termed as “Big Data”. 
Many tools and frameworks have been developed to assess faster processing of such large amount of data. HIVE is one such data warehouse software project built on top of Apache Hadoop for providing data summarization, query and analysis. It gives an SQL-like interface to query data stored in various databases and file systems that integrate with Hadoop. While it has been developed by Facebook, it is used and developed by other companies such as Netflix and Amazon. It eventually converts all the queries to MapReduce jobs. 

 The dataset used in this project has been obtained from the UCI Repository is the health care data of Americans based on attributes (18) such as Eating Habits, Drinking Habits, Sleeping Patterns, working hours, weight, height, personality and others. We have preprocessed the data using the WEKA tool. Few machine learning algorithms have been applied on the dataset for the predictive analysis to determine whether a person is healthy or not.

# Objectives

The main objective of my project is to predict whether a person is healthy or not based on some parameters using machine learning algorithms and I have used HIVE for the storage of the large dataset which is not feasible in traditional data warehouses. I found which attributes are strongly associated to each other based on a association algorithm, i.e Apriori Algorithm. I divided data sets into parts- training and testing set and found which algorithm alone and in combination gives us the best accuracy. 

# Implementation 

On the dataset, I performed various operations such as classification, clustering and association. For classification, I applied SVM (Support Vector Machine) and Decision tree algorithms. For clustering of datasets, I applied k-means clustering. For determining strong association between the attributes or to determine the attributes whose confidence level are above the threshold level, I applied the Apriori association algorithm. The attributes or features that I obtained after implementing Apriori algorithm are then used to build the decision tree for the classifying the records in the training set. 

# Python Libraries

Following are the libraries used for this project:-

• pyhs2

• itertools

• random

• numpy

• pandas

• urllib3

• sklearn

• pygal

• thrift

# Running the project

Clone the repository to create a local copy on your computer.

Run the python file (health.py) in the terminal using the following command:
                python health.py
