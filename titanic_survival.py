# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:07:22 2017

@author: Pavlos Sakoglou

Course: Machine Learning Nanodegree, Udacity
"""

# Importing pandas -- no need for more imports 
import pandas as pd

# Get data from csv and store them in a vector and matrix: the vector with 
# all the observations (survived) and the matrix of all data
df = pd.read_csv('titanic_data.csv', delimiter=',')

outcomes = df['Survived']
data = df.drop('Survived', axis = 1)

# Given function that returns a string message with the accuracy score of input
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"

'''
# Question 1
'''
def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        
        # Predict the survival of 'passenger'
        # We append 0, meaning that they all died. Otherwise we append 1
        # i.e. 0 := Died, 1 := Survived
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Initialize predictions vector as above, and compute the accuracy
predictions = predictions_0(data)
print("\nQuestion 1:")
print(accuracy_score(outcomes, predictions))
print("\n")
'''
#
# Answer 1: predictions have an accuracy of 61.62%
#
#####################################################
'''

'''
# Question 2
'''
def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female' :
            predictions.append(1)
        else :    
            predictions.append(0)
        
    
    # Return our predictions
    return pd.Series(predictions)

# Initialize predictions vector as above, and compute the accuracy
predictions = predictions_1(data)
print("Question 2:")
print(accuracy_score(outcomes, predictions))
print("\n")
'''
#
# Answer 2: predictions have an accuracy of 78.68%
#
#####################################################
'''


'''
# Question 3
'''
def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female' or (passenger['Sex'] == 'male' and passenger['Age'] < 10) :
            predictions.append(1)
        else :    
            predictions.append(0)
        
    
    # Return our predictions
    return pd.Series(predictions)

# Initialize predictions vector as above, and compute the accuracy
predictions = predictions_2(data)
print("Question 3:")
print(accuracy_score(outcomes, predictions))
print("\n")
'''
#
# Answer 3: predictions have an accuracy of 79.35%
#
#####################################################
'''

'''
# Question 4
'''
def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # If you are female
        if passenger['Sex'] == 'female':
            # Parametrize the females further
            # If you are between 40 and 50 and 3rd class, predict death
            if passenger['Age'] > 40 and passenger['Age'] < 50 and passenger['Pclass'] > 2:
                predictions.append(0)
            else:
                # else predict survival
                predictions.append(1)
        # If you are male
        else:
            # If you are first class and under 40, or under 8 in general, predict survival
            if (passenger['Pclass'] == 1 and passenger['Age'] < 40) or passenger['Age'] < 8 :
                predictions.append(1)
            else:
                # else predict death
                predictions.append(0)
        
    # Return our predictions
    return pd.Series(predictions)

# Initialize predictions vector as above, and compute the accuracy
predictions = predictions_3(data)
print("Question 4:")
print(accuracy_score(outcomes, predictions))
print("\nQuestion 5:\nSee comments!\n")
'''
#
#
# Answer 4: predictions have an accuracy of 80.25%
#
# Steps: In the most part, the result was derived by trial and error.
#        Then, recording my observations from my tests on various parameters
#        and their values, I came up with a logical plan that would increase
#        the probability of survival of certain groups, and implemented them 
#        as conditions.
#
#       I looked at Age, Gender, and Pclass at the most part, as these features
#       would mostly increase the probabilities. I also noticed that Sex == female
#       needed extra parametrization, cause the probability of survival of females
#       was already contributing much in the result. Thus instead of trying to increase
#       by adding favorable (pressumably) parameters, I decided to add unfavorable parameters
#       and instead record them as probability of death -- exclude them from the value space. 
#
#       The conditions as shown above were to split the population in 2: male and female, 
#       since this was the obvious largest deterministic factor of survival. Then, I looked 
#       into ages and class for both groups, and exluded (or included) the most likely events 
#       for death or survival, respectively. 
#
#       My predictions are 80.25% accurate. By keep parametrizing, I would probably
#       hit higher accuracy. 
#
#####################################################
'''


'''
#
#
# Answer 5:
#
#       In the titanic case, the y variable (observation) was the "survived" column of data. 
#       The independent variables were the parameters I picked, i.e. x_1 = Age, x_2 = Sex, etc. 
#
#       Another scenario would be to predict if a candidate will get a job, given a data set of 
#       N observed results of N candidates. That is, we are given that N candidates with certain 
#       characteristics (Test scores, GPA, Resume rank in scale of 1-10, interview score, ethnicity, etc.) 
#       have either passed or not (1 = got the job, 0 == not gotten the job), and our task is to predict 
#       if the (N+1)th candidate will get the job or not, and with what probability (accuracy of our model). 
#
#       Then the independent variables would be the parameters of the N past candidates, 
#       i.e. resume rank, gpa etc. and the y dependent variable (observed for each candidate) 
#       would be the outcome. Then we would try to set parameters as:
#       if (candidate['GPA'] > 3.2 and candidate['Resume'] > 0.7) : predictions.append(1)
#       else : predictions.append(0)
#
#       etc. Then, similarly we would run: print(accuracy_score(outcomes, predictions))
#       with outcomes having the y_i variables (0 or 1 for each past candidate), and predictions
#       generated as above. 
#
#       I would guess that the two most common features would be the interview score, and a test score,
#       i.e. a special test that each candidate have to take to assess their skills on a subject. 
#
#####################################################
'''


