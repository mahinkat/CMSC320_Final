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

## Data Collection Process
The first step of the data science lifecycle involves either gathering or finding gathered data that is relevant to our topic. Because it would take an unrealistic amount of time to gather this information ourselves, we opted to perform secondary data collection, which involves using information already gathered by others. An important consideration we abided by was ensuring that the data originated from a reputable and reliable source, which we were able to find on Kaggle and verify with the NBA's own sources.

We downloaded the NBA player data and performance statistics from 2012 to 2023 into a CSV file format, mounted it to the content drive, and loaded it into the project directory. From there, we filtered out the data that was unnecessary or would create noise for our analysis. Performing this first data collection step was a critical component of the data science testing process, as it allowed the following exploratory data analysis and hypothesis testing to run smoothly and accurately.

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

![Proportion of Player to Seasons by Team](https://github.com/mahinkat/CMSC320_Final/blob/main/img1320.png?raw=true)

After cleaning up this dataset, we can see that this data set is very robust set that has uniformity with unique player season entries. This will allow us to pick and choose a healthy variety of ways to view the data and analyze it due to the large amounts of data provided.

One fear I had going into this data set is that the teams would not be represented fairly, with some overpowering the others. After analyzing the statistics of team counts, it seems they all ballpark around the same amount as the standard deviation is a relatively low number. The only outlier we can see is NOH, and this is due to the relocation of a franchise from New Orleans to Charlotte around 2011/2012 but is easy to deal with as we can either cut that data or add it to the Charlotte Hornets.

**Summary:**
The analysis reveals relatively balanced player distribution across all 31 NBA teams represented in the dataset. On average, each team accounts for approximately 3.23% of the total player-seasons, with a standard deviation of 0.56%. The minimum proportion is 0.27% and the maximum is 3.69%, indicating that while most teams have similar representation, there is some variation. This balanced distribution is important for ensuring that our subsequent analyses and clustering algorithms are not biased toward any particular team.

---

---

## Exploratory Data Analysis (EDA)

We formulated several hypotheses which we were curious about and ran what we felt were the most appropriate tests regarding each hypothesis. In this part of the data science process, we will run various types of hypothesis testing to verify whether or not our standard or null hypothesis should be rejected or not. To determine whether or not to reject a null hypothesis, we calculate a p-value and compare it against a pre-determined significance level, ɑ.

### 1. Hypothesis Testing: Chi-Squared Test on Shooting Efficiency and Player Court Time

**Research Question:** Let's examine the data to determine whether shooting efficiency (measured by FG_PCT) is related to a player's position/role on the court. Does a player's playing time correlate to their shooting efficiency?

**Hypotheses:**
- **Null Hypothesis (H₀):** A player's shooting efficiency is independent of their playing time.
- **Alternative Hypothesis (H₁):** A player's shooting efficiency is NOT independent of their playing time.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Categorize players by minutes played
min_threshold = df['MIN'].median()
df['PLAYER_ROLE'] = df['MIN'].apply(lambda x: 'High Minutes (Starter)' if x >= min_threshold else 'Low Minutes (Bench)')

# Categorize players by shooting efficiency
fg_threshold = df['FG_PCT'].median()
df['SHOOTING_EFFICIENCY'] = df['FG_PCT'].apply(lambda x: 'Efficient Shooter' if x >= fg_threshold else 'Inefficient Shooter')

print(f"\nMinutes Threshold (Median): {min_threshold:.2f}")
print(f"Field Goal % Threshold (Median): {fg_threshold:.3f}")

# Create contingency table
contingency_table = pd.crosstab(
    df['PLAYER_ROLE'],
    df['SHOOTING_EFFICIENCY'],
    margins=True
)
print("Contingency Table with Margins:")
print(contingency_table)

# Perform chi-squared test
chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)

print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"P-value: {p_value}")

print("\nExpected Frequencies:")
expected_df = pd.DataFrame(
    expected_freq,
    index=contingency_table.index,
    columns=contingency_table.columns
)
print(expected_df)
```

**Output:**
```
Minutes Threshold (Median): 1037.00
Field Goal % Threshold (Median): 0.441
Contingency Table with Margins:
SHOOTING_EFFICIENCY     Efficient Shooter  Inefficient Shooter   All
PLAYER_ROLE                                                         
High Minutes (Starter)               1882                 1248  3130
Low Minutes (Bench)                  1273                 1856  3129
All                                  3155                 3104  6259

Chi-Square Statistic: 236.6460
Degrees of Freedom: 4
P-value: 4.894295649695702e-50

Expected Frequencies:
SHOOTING_EFFICIENCY     Efficient Shooter  Inefficient Shooter     All
PLAYER_ROLE                                                           
High Minutes (Starter)        1577.752037          1552.247963  3130.0
Low Minutes (Bench)           1577.247963          1551.752037  3129.0
All                           3155.000000          3104.000000  6259.0
```

**Conclusion:** From this result, we can see that the p-value (4.89 × 10⁻⁵⁰) is much less than 0.05, so we reject the null hypothesis. We can conclude that there is a statistically significant relationship between playing time and shooting efficiency. Thus, whether a player gets high or low minutes is NOT independent of their shooting efficiency. The contingency table shows that players with high minutes (starters) are more likely to be efficient shooters (1,882 efficient vs 1,248 inefficient), while bench players show the opposite pattern (1,273 efficient vs 1,856 inefficient).
```python
# Visualize the relationship
plt.figure(figsize=(10, 6))
contingency_table.plot(kind='bar', stacked=True, color=['red', 'blue'])
plt.title('Shooting Efficiency by Playing Time')
plt.xlabel('Player Role')
plt.ylabel('Number of Players')
plt.xticks(rotation=0)
plt.legend(title='Shooting Efficiency')
plt.tight_layout()
plt.show()
```

**Visualization:**

![Shooting Efficiency by Playing Time](https://github.com/mahinkat/CMSC320_Final/blob/main/img2320.png?raw=true)

**Key Insights:**
- The chi-squared statistic of 236.65 with 4 degrees of freedom indicates a very strong association between the two variables
- Starters (high minutes) tend to be more efficient shooters, which makes logical sense as coaches typically give more playing time to players who perform better
- This finding could be useful for fantasy basketball decisions and understanding player value

---

### 2. Hypothesis Testing: ANOVA on Points Scored by Team

**Research Question:** To investigate if there are statistically significant differences in the average points scored per player across different teams, we will perform a one-way ANOVA test.

**Hypotheses:**
- **Null Hypothesis (H₀):** There is no statistically significant difference in the mean points scored per player among all the different teams.
- **Alternative Hypothesis (H₁):** There is a statistically significant difference in the mean points scored per player for at least one team compared to the others.

```python
import scipy.stats as stats

# Group points data by team
team_points = [df[df['TEAM'] == team]['PTS'] for team in df['TEAM'].unique()]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(*team_points)

print(f"ANOVA F-statistic: {f_statistic:.4f}")
print(f"ANOVA p-value: {p_value:.4f}")
```

**Output:**
```
ANOVA F-statistic: 0.6732
ANOVA p-value: 0.9110
```

**Conclusion:** The p-value (0.9110) is much greater than the significance level (α = 0.05), so we fail to reject the null hypothesis. This suggests that there is no statistically significant difference in the mean points scored among different teams. The average points scored per player across teams remains relatively uniformly distributed. This finding indicates that team affiliation does not significantly impact individual player scoring averages, suggesting that scoring opportunities and player performance are fairly consistent across the league.
```python
# Visualize the distribution of points by team
plt.figure(figsize=(15, 8))
df.boxplot(column='PTS', by='TEAM', rot=90)
plt.title('Distribution of Points Scored per Player by Team')
plt.xlabel('Team')
plt.ylabel('Points Scored (PTS)')
plt.suptitle('')
plt.tight_layout()
plt.show()
```

**Visualization:**

![Distribution of Points Scored per Player by Team](https://github.com/mahinkat/CMSC320_Final/blob/main/cmsc3320.png?raw=true)

**Key Insights:**
- The F-statistic of 0.6732 is relatively small, indicating minimal variance between team means compared to variance within teams
- The extremely high p-value (0.9110) provides strong evidence that any observed differences in mean points across teams are due to random chance
- The box plot visualization confirms this finding, showing similar distributions of player points across all 31 teams
- This uniformity suggests that the NBA maintains competitive balance, with no team systematically producing higher or lower-scoring players
- For fantasy basketball and betting purposes, this indicates that team selection should not be a primary factor when evaluating a player's scoring potential

---
