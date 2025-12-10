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

**Dataset Summary:**
- **Total Entries:** 6,259
- **Total Features:** 29
- **Memory Usage:** 1.4+ MB

**Feature Breakdown:**
- 20 integer features (player IDs, statistics counts)
- 5 float features (percentages and ratios)
- 4 object features (year, season type, player name, team)

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

**Team Distribution Statistics:**
- **Count:** 31 teams
- **Mean proportion:** 0.032258 (approximately 3.23% per team)
- **Standard deviation:** 0.005623
- **Range:** 0.002716 to 0.036907

The analysis shows relatively balanced player distribution across all 31 NBA teams, with most teams having between 3.2% and 3.4% of the total player-seasons in the dataset.

---
