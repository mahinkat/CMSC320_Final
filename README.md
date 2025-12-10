# Player Archetype Analysis
**Fall 2025 Data Science Project**  
**Team Members:** Jayden L., Richeek T., Nathan H., Mahin K. Owen, Alex Y.

---

## Contributions
**A: Project Idea:**  
Richeek, Jayden, Nathan, and Mahin decided on analyzing an NBA dataset.

**B: Dataset Curation and Preprocessing:**  
Richeek and Jayden converted the dataset into a Pandas dataframe, filtered empty rows, and dropped irrelevant columns.

**C: Data Exploration and Summary Statistics:**  
Richeek analyzed simple statistics. Nathan conducted a chi-squared test on shooting efficiency and player court time. Owen did an ANOVA on Points Scored by Team. Mahin did a Z-Test for High-Defense Players Points.

**D: ML Algorithm Design/Development:**  
Richeek and Jayden used the elbow method for K-Means clustering on points, assists, steals, rebounds, blocks, and field goal percentage.

**E: ML Algorithm Training and Test Data Analysis:**  
Nathan and Alex reviewed models and data. Nathan performed testing and analysis for K-Means clustering.

**F: Visualization, Result Analysis, Conclusion:**  
Richeek and Jayden visualized the Elbow method and clustering results. Nathan analyzed and visualized clusters.

**G: Final Tutorial Report Creation:**  
Jayden wrote the Contributions and Parts of the Introduction.

---

## Introduction
The purpose of this project is to learn and apply the full data science testing procedure using large, real-world datasets. To achieve this goal, we analyzed a dataset from the National Basketball Association (NBA), a multibillion-dollar industry with millions of spectators each game. Basketball continues to remain one of the most popular sports in the world, with an estimated 2.4 billion fans globally, and the NBA is often widely regarded as the premier league within the sport. Our data set was on traditional box score statistics for players in the National Basketball Association during the 2012 to 2023 regular season. This includes details about each player from this 100-year span, such as the team they play for, the number of games played, minutes on court, field-goals attempted, field-goals made, and assists.



---

## Importance
As one of the most-watched sports in the world, there is near constant speculation amongst the basketball community regarding which players and teams are favored to win games, championships, and awards. Fans bet and wager billions of dollars annually on their preferred teams and players in hopes of making money, and as the sport continues to grow in popularity worldwide, it may be of interest to some to understand who the best and worst players are. By using data science testing techniques, we can predict which NBA players are likely to earn awards and titles such as MVP and All-Star, so fantasy and basketball fans can make informed decisions when choosing their rosters.

---

## Data Import and Statistical Methods

### Loading the Dataset
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Output:**
```
Mounted at /content/drive
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/drive/MyDrive/CMSC320 Final Portfolio/nba_data/Regular_Season.csv')
```

### Statistical Analysis

#### 1. Dataset Overview: Features, Entries, and Data Types
```python
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
total_entries = df.shape[0]
total_features = df.shape[1]

print(f"Total Entries: {total_entries}")
print(f"Total Features: {total_features}")
print("\nFeature Names and Data Types:")
print(df.info())
```

**Output:**
```
Total Entries: 6259
Total Features: 29

Feature Names and Data Types:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6259 entries, 0 to 6258
Data columns (total 29 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   year         6259 non-null   object 
 1   Season_type  6259 non-null   object 
 2   PLAYER_ID    6259 non-null   int64  
 3   RANK         6259 non-null   int64  
 4   PLAYER       6259 non-null   object 
 5   TEAM_ID      6259 non-null   int64  
 6   TEAM         6259 non-null   object 
 7   GP           6259 non-null   int64  
 8   MIN          6259 non-null   int64  
 9   FGM          6259 non-null   int64  
 10  FGA          6259 non-null   int64  
 11  FG_PCT       6259 non-null   float64
 12  FG3M         6259 non-null   int64  
 13  FG3A         6259 non-null   int64  
 14  FG3_PCT      6259 non-null   float64
 15  FTM          6259 non-null   int64  
 16  FTA          6259 non-null   int64  
 17  FT_PCT       6259 non-null   float64
 18  OREB         6259 non-null   int64  
 19  DREB         6259 non-null   int64  
 20  REB          6259 non-null   int64  
 21  AST          6259 non-null   int64  
 22  STL          6259 non-null   int64  
 23  BLK          6259 non-null   int64  
 24  TOV          6259 non-null   int64  
 25  PF           6259 non-null   int64  
 26  PTS          6259 non-null   int64  
 27  AST_TOV      6259 non-null   float64
 28  STL_TOV      6259 non-null   float64
dtypes: float64(5), int64(20), object(4)
memory usage: 1.4+ MB
None
```

**Summary:**
The dataset contains 6,259 player-season entries with 29 features covering the 2012-2023 NBA regular seasons. All columns have complete data (no null values). The features include player identification, team information, and comprehensive box score statistics such as games played (GP), minutes (MIN), field goals made/attempted (FGM/FGA), three-pointers, free throws, rebounds, assists, steals, blocks, turnovers, personal fouls, and points.

#### 2. Player Distribution by Team
```python
team_counts = df['TEAM'].value_counts(normalize=True).sort_values(ascending=False)

print("\nTeam Counts Statistics:")
print(team_counts.describe())

plt.figure(figsize=(15, 6))
sns.barplot(x=team_counts.index, y=team_counts.values)
plt.title('Proportion of Player to Seasons by Team', fontsize=14)
plt.xlabel('Team Abbreviation', fontsize=12)
plt.ylabel('Proportion of Dataset', fontsize=12)
plt.xticks(rotation=90, ha='center', fontsize=10)
plt.tight_layout()
plt.show()
```

**Output:**
```
Team Counts Statistics:
count    31.000000
mean      0.032258
std       0.005623
min       0.002716
25%       0.032433
50%       0.033072
75%       0.033791
max       0.036907
Name: proportion, dtype: float64
```

![Proportion of Player to Seasons by Team](visualization-placeholder)

**Summary:**
The analysis reveals relatively balanced player distribution across all 31 NBA teams represented in the dataset. On average, each team accounts for approximately 3.23% of the total player-seasons, with a standard deviation of 0.56%. The minimum proportion is 0.27% and the maximum is 3.69%, indicating that while most teams have similar representation, there is some variation. This balanced distribution is important for ensuring that our subsequent analyses and clustering algorithms are not biased toward any particular team.

---
