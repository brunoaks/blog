---
layout: post
title: Predicting the 2018 FIFA World Cup games
---

I just finished my first personal project to practice data mining and Machine Learning concepts, and I decided to write a post telling a bit more about the development of the project and the results I achieved. 

I highly encourage you to access my [Github repository](https://github.com/brunoaks/FIFA-worldcup-2018-prediction) so that you can open the Jupyter Notebook with all the code and step-by-step explanations. The Jupyter Notebook contains a little more data exploration, but I condensed the main code in this post to make it clearer and more objective.

### 1. Defining the project
The idea was simple: **to create an ML model capable of predicting the outcomes of the 2018 FIFA World Cup matches**. The model would be trained on historical data of past international matches. I followed the general workflow of a Data Science project: _initial planning, data extraction, data mining and cleaning, model selection, model deployment and analysis_. 

In a bigger project, potentially with a real client, a lot of validation and pivoting would be necessary after the model deployment - some features would have to be changed, some parameters tuned, or maybe an entirely different prediction model would have to be chosen. I left these steps as "Next Steps" in my Jupyter Notebook.

In this post, I will write about all the procedures in the order I followed them, so you can understand a little better the rationale behind the lines of code.

### 2. Initial planning
Before starting to hunt for data, I thought about all the variables that could be useful in this kind of analysis. It is essentially a classification problem: given a set of values in different columns (features), predict whether Team A will win, Team B or if it will be a tie. I would need a database containing lots of international matches' outcomes, and I thought of some variables that this database could have in order to make our analysis more interesting:

* Weather conditions;
* Match location;
* Strength of each team as average of each player's rating;
* Each team's track record in international matches up until that game;
* FIFA ranking position of each team at the time of the game;
* etc.

With that in mind, I set out to gather some data.

### 3. Data Extraction
I had a lot of difficulty finding a dataset that wold give me not only international matches' outcomes, but also the variables I mentioned before. For example, assessing each team's strength proved to be a daunting task. I thought about using the players' ratings in each team extracted from the FIFA game, but it was released in the 90s, and much of my dataset would have blank values. 

After searching extensively for good datasets, I decided to stick to a neat [dataset from Kaggle](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017) containing about 40,000 international football matches' results. It would not have all the features I wanted, but it would be a good start.

![wc1](https://brunoaks.github.io/images/wc1.JPG)

### 4. Data Mining and Cleaning
After playing around with the data, I decided I would skim down the dataset to its bare essentials: just the name of the teams and the outcome of the match. This way, the model would not be confused by other variables that could potentially harm our accuracy, like match location (there would be an winning inclination to teams who won in matches played in Russia) or championship (there would be bias against games other than World Cup matches).

I used Python's **pandas** library for doing the data manipulation. The end result is a dataset containing only games with the participants of the 2018 World Cup (we don't really care about matches with teams who are not participating in the championship) and only from 1930 onwards (when the first World Cup was held and football started getting competitive). 

You can check out the code below.

```python
# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read .csv files
results = pd.read_csv('datasets/results.csv')

# Create a DF with all participating teams
wc_teams = ['Australia', ' Iran', 'Japan', 'Korea Republic', 
            'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria', 
            'Senegal', 'Tunisia', 'Costa Rica', 'Mexico', 
            'Panama', 'Argentina', 'Brazil', 'Colombia', 
            'Peru', 'Uruguay', 'Belgium', 'Croatia', 
            'Denmark', 'England', 'France', 'Germany', 
            'Iceland', 'Poland', 'Portugal', 'Russia', 
            'Serbia', 'Spain', 'Sweden', 'Switzerland']

# Filter the 'results' dataframe to show only teams in this year's world cup
df_teams_home = results[results['home_team'].isin(wc_teams)]
df_teams_away = results[results['away_team'].isin(wc_teams)]
df_teams = pd.concat((df_teams_home, df_teams_away))
df_teams.drop_duplicates()

# Loop for creating a new column 'year' in the DataFrame
year = []
for row in df_teams['date']:
    year.append(int(row[:4]))
df_teams['match_year'] = year

# Slice the dataset with matches that took place from 1930 onwards (the year of the first ever World Cup)
df_teams30 = df_teams[df_teams.match_year >= 1930]

# Drop all columns we don't need
df_teams30 = df_teams30.drop(['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'goal_difference', 'match_year'], axis=1)

# Convert match outcomes to numeric values ('2' for home team win, '1' for tie and '0' for away team win)
df_teams30 = df_teams30.reset_index(drop=True)
df_teams30.loc[df_teams30.winning_team == df_teams30.home_team, 'winning_team']= 2
df_teams30.loc[df_teams30.winning_team == 'Tie', 'winning_team']= 1
df_teams30.loc[df_teams30.winning_team == df_teams30.away_team, 'winning_team']= 0
```

If we call `df_teams30.head()` we can assert that the table is in a format that suits us:


![wc2](https://brunoaks.github.io/images/wc2.JPG)

Now, before we initialize a Machine Learning model, I just have to first do some one-hot encoding and then split the dataset in train/test chunks. 

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get dummy variables
final = pd.get_dummies(df_teams30, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# Separate X and y sets
X = final.drop(['winning_team'], axis=1)
y = final["winning_team"]
y = y.astype('int')

# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```


### 5. Model selection
I first tried a simple Logistic Regression model on the dataset using Python's sci-kit learn.

```python
# Elaborate Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))
```
As a result, I got the following:

`Training set accuracy: 0.571`

`Test set accuracy: 0.555`

Which is better than throwing darts blindfolded at the three possible outcomes, but still not outstanding. I tried testing a bunch of other models with the following block of code, but it didn't seem to work much.

```python
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

acc_dict = {}

# Loop to do fit and predictions of each classifier into the dataset
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    
    # Storing each score into a dict
    if name in acc_dict:
        acc_dict[name] += acc
    else:
        acc_dict[name] = acc

# Storing the results in a DataFrame to be visualized
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
```
We can observe that the Logistic Regression stands out as the classifier with the best accuracy out of all. 

![wc3](https://brunoaks.github.io/images/wc3.png)

That way, I decided to keep the Logistic Regression model and use it for the rest of the project as my main classifier.

### 6. Model deployment and analysis
With the model already trained and instantiated, we just need a prediction dataset in which we'll apply the model. It needed to have the same features as the training and test data, so I had to do some data preprocessing.

Also, one small problem I ran into was that in the training set, the teams in each match were separated into "Home" and "Away" teams. However, in World Cup matches, there is no such division, since all teams but the host are playing "away" from home. Therefore, I needed a criterium to separate the teams into these two categories. I decided to get a dataset containing the FIFA ranking position of each nation, and based on this dataset, I would set the team with the higher position as the "home" team, because he could be considered the most likely to win the match. 

_Disclaimer: It's a practical solution, not a fundamentally correct one. In practical terms, the model was naturally favouring against teams playing at home because they have a natural advantage according to the training set; however, we have small teams such as Iceland that have played very few international matches, and they may hold excellent track records, but they're much less likely to win a match against some of the world's strongest teams, such as Germany. This was a way to give an arbitrary edge to the stronger team, with information not available previously._

Now, onto the data preprocessing. For the group stage games, I had the help of a little dataset called `fixtures` obtained from [this website](https://fixturedownload.com/results/fifa-world-cup-2018). So the steps I followed in pseudocode were:
```
For each match in fixtures
    Get teams' FIFA ranking position
    If (team 1 position) < (team 2 position)
        set team 1 to home team
    Else
        set team 2 to home team
Set dummy variables
Ensure same number columns as the training dataset
```

Then, with the prediction set neatly arranged, I could deploy the model. For the group stage games I did it in a sort of experimental way, as you can see in the Jupyter Notebook, but for the other games I created a function for making the whole process easier, from data cleaning to model deployment, which you can check out below.

```python
def clean_and_predict(matches, ranking, training_set, logreg):

    """
    Cleans data and predicts matches' outcomes

    Arguments:
    matches -- a list of tuples containing in each tuple the two teams which are playing in a match
    ranking -- the FIFA ranking dataset
    training_set -- the dataset used for training the data
    logreg -- our logistic regression model

    """

    # Initialization of auxiliary list for data cleaning
    positions = []

    # Loop to retrieve each team's position according to FIFA ranking
    for match in matches:
        positions.append(ranking.loc[ranking['Team'] == match[0],'Position'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == match[1],'Position'].iloc[0])
    
    # Creating the DataFrame for prediction
    pred_set = []

    # Initializing iterators for while loop
    i = 0
    j = 0

    # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
    while i < len(positions):
        dict1 = {}

        # If position of first team is better, he will be the 'home' team, and vice-versa
        if positions[i] < positions[i + 1]:
            dict1.update({'home_team': matches[j][0], 'away_team': matches[j][1]})
        else:
            dict1.update({'home_team': matches[j][1], 'away_team': matches[j][0]})

        # Append updated dictionary to the list, that will later be converted into a DataFrame
        pred_set.append(dict1)
        i += 2
        j += 1

    # Convert list into DataFrame
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    # Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

    # Add missing columns compared to the model's training dataset
    missing_cols2 = set(training_set.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[training_set.columns]

    # Remove winning team column
    pred_set = pred_set.drop(['winning_team'], axis=1)

    # Predict!
    predictions = logreg.predict(pred_set)
    for i in range(len(pred_set)):
        print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
        if predictions[i] == 2:
            print("Winner: " + backup_pred_set.iloc[i, 1])
        elif predictions[i] == 1:
            print("Tie")
        elif predictions[i] == 0:
            print("Winner: " + backup_pred_set.iloc[i, 0])
        print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ' , '%.3f'%(logreg.predict_proba(pred_set)[i][2]))
        print('Probability of Tie: ', '%.3f'%(logreg.predict_proba(pred_set)[i][1])) 
        print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ', '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
        print("")
```

After calling the functon, you get your predictions printed out in an organized way, with the probabilities of each team winning the match.

I'll put here the results for the knockout stage here, just for curiosity's sake.

**Round of 16**

![wc5](https://brunoaks.github.io/images/wc5.jpg)

**Quarter-finals, Semi-Finals and Finals**

![wc6](https://brunoaks.github.io/images/wc6.jpg)

Germany's definitely the favourite for this World Cup!

### 7. Conclusion
That's about it! Thanks for reading until the end. In the Jupyter Notebooks I highlighted some possible next steps that could be made. Keep reading the blog for more details on new projects!