# Player Archetype Analysis
**Fall 2025 Data Science Project**  
**Team Members:** Jayden L., Richeek T., Nathan H., Mahin K. Owen S., Alex Y.

---

## Contributions
**A: Project Idea:**  
Richeek, Jayden, Nathan, and Mahin decided on analyzing an NBA dataset.

**B: Dataset Curation and Preprocessing:**  
Richeek and Jayden converted the dataset into a Pandas dataframe, filtered empty rows, and dropped irrelevant columns.

**C: Data Exploration and Summary Statistics:**  
Richeek analyzed simple statistics about the data and their columns. Nathan conducted a chi-Sqaured Test on shooting efficiency and player court time. Owen did an ANOVA on Points Scored by Team. Mahin did the Z-Test for High-Defense Players Points.

**D: ML Algorithm Design/Development:**  
Richeek and Jayden used the elbow method to determine the best k-value to use for K-Means clustering and then handled the K-means cluster modeling using the points, assists, steals, rebounds, blocks, and field goal percentage features.

**E: ML Algorithm Training and Test Data Analysis:**  
Nathan and Alex reviewed and tested all ML models, data, and hypothesis testing to ensure correct functionality. Nathan performed data testing and analysis for the K-Means clustering.

**F: Visualization, Result Analysis, Conclusion:**  
Richeek and Jayden visualized the Elbow method for finding the best k-value for K-Mean Clusteting and the result of the K-Means Clustering. Nathan performed result analysis for the K-Means clustering,  calculating a clustering evaluation metric and visualizing the clusters.

**G: Final Tutorial Report Creation:**  
Jayden wrote the Contributions and Parts of the Introduction, Owen wrote most of introduction and introductions for each step of data science process. Mahin wrote/transferred everything to Github Pages, added key insights to parts of the file, and did the conclusion. 

**H: Additional 

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

### 3. Hypothesis Testing: Z-Test for High-Defense Players Points

**Research Question:** We'll test whether players with high defensive contributions (steals + blocks) score fewer points than the overall player average.

**Hypotheses:**
- **Null Hypothesis (H₀):** The mean points scored by high-defense players is equal to the overall mean points.
- **Alternative Hypothesis (H₁):** The mean points scored by high-defense players is less than the overall average.

```python
import numpy as np
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate overall population mean points
overall_mean = df['PTS'].mean()

# Define defensive score as STL + BLK
df['DEF_SCORE'] = df['STL'] + df['BLK']

# Identify top 25% defensive players
threshold = np.percentile(df['DEF_SCORE'], 75)
high_def_pts = df[df['DEF_SCORE'] >= threshold]['PTS']

# Perform one-sample z-test
z_stat, p_value = ztest(high_def_pts, value=overall_mean, alternative='smaller')

# Significance level
alpha = 0.05

# Visualize High-Defense players vs All players
plt.figure(figsize=(10,6))
sns.histplot(df['PTS'], bins=30, kde=True, color='gray', label='All Players')
sns.histplot(high_def_pts, bins=15, kde=True, color='green', label='Top 25% Defense')
plt.axvline(overall_mean, color='blue', linestyle='--', label='Overall Mean')
plt.title("Points Distribution: High-Defense Players vs All Players")
plt.xlabel("Points (PTS)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Overall mean points: {overall_mean:.2f}")
```

**Output:**
```
Z-statistic: 44.0895
P-value: 1.0000
Overall mean points: 494.86
```

**Visualization:**

![Points Distribution: High-Defense Players vs All Players](https://github.com/mahinkat/CMSC320_Final/blob/main/cmsc4.png?raw=true)

**Conclusion:** From the graph and the p-value (1.0000), which is much greater than α = 0.05, we fail to reject the null hypothesis. High-defense players do not score significantly fewer points than the overall average. In fact, the extremely high positive z-statistic (44.0895) and p-value of 1.0000 indicate that high-defense players actually score MORE points on average than the overall population, which is the opposite of what we hypothesized.

**Key Insights:**
- The z-statistic of 44.09 is extraordinarily high and positive, suggesting that defensive prowess and scoring ability are positively correlated rather than negatively correlated
- This counterintuitive finding challenges the common assumption that players specialize in either offense or defense
- The visualization shows that the distribution of points for high-defense players (green) is shifted to the right compared to all players (gray), confirming they score more points
- This suggests that elite NBA players tend to excel in multiple aspects of the game, being strong on both offense and defense
- For fantasy basketball and team building, this indicates that high-defensive players are valuable not just for their defensive stats but also for their offensive contributions
- The result makes basketball sense: players who get more playing time (typically starters) have more opportunities to accumulate both defensive stats (steals and blocks) and points

---

## Primary Analysis

In this step of the data science process, we aim to build and train a predictive model to help us learn and predict what a player's archetype is likely to be. We chose to perform the Elbow Method, which involves optimal clustering in K-Means. Our objective is to see whether we can predict what role a player is likely to take on given their historical performance statistics.

### Import Libraries and Prepare Data

First, let's import essential libraries and re-add/clean the data to have a fresh base to start with.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Tests
from scipy.stats import chi2_contingency
import scipy.stats as stats
from statsmodels.stats.weightstats import ztest

# K-Elbow Visualization
from scipy.spatial.distance import cdist

# Classification and Clustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load and clean data
df = pd.read_csv('/content/drive/MyDrive/CMSC320 Final Portfolio/nba_data/Regular_Season.csv')
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
```

### Elbow Method and Standardization

```python
# --- K-Means: Determining the Optimal Number of Clusters (K) ---

# 1. Select and Standardize Features
clustering_features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT']
X_cluster = df[clustering_features]
seed = 0

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Apply Principal Component Analysis and return the transformed data
def apply_pca(X_scaled, n_components):
    # Fit the PCA model
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    
    return pca, X_pca

# Apply PCA
n_components = 4
pca, X_pca = apply_pca(X_scaled, n_components)

# 3. Visualizing the Elbow
# Store distortion values
distortions = []
K = range(1, 10)
# Hold key-pair values for k values and their respective distortion value
kmap = {}

# Loop through each K value from 1-10
for k in K:
    # Create a K-Mean model and fit it using X_pca data
    k_mean = KMeans(n_clusters=k, random_state=seed)
    k_mean.fit(X_pca)
    
    # Calculate the mean distortion for each K-value and add it to the map
    euclidean_dist = cdist(X_pca, k_mean.cluster_centers_, "Euclidean") ** 2
    k_dist = sum(np.min(euclidean_dist, axis=1)) / X_pca.shape[0]
    distortions.append(k_dist)
    kmap[k] = k_dist

# Plot the elbow curve
plt.plot(K, distortions, 'bx-')
plt.xlabel('K-values')
plt.ylabel('Distortion')
plt.title('The Elbow Method for K-Mean Clustering')
plt.show()
```

**Visualization:**

![The Elbow Method for K-Mean Clustering](https://github.com/mahinkat/CMSC320_Final/blob/main/cmsc5.png?raw=true)

**Interpretation:** This plot helps us determine the best size of K for the clustering analysis. We chose K=4 as we can see that is where the bend begins (the "elbow"). At this point, adding more clusters doesn't significantly reduce the distortion, indicating that 4 clusters provide a good balance between model complexity and clustering quality. The elbow represents the point of diminishing returns in variance explanation.

### K-Means Clustering

Here we are going to apply the K-Means clustering and analyze the four generated clusters into statistics relating to the data set to see the similarities.

```python
# Apply K-Means Clustering (K=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze Cluster Centers
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
cluster_summary = pd.DataFrame(cluster_centers_original, columns=clustering_features)
cluster_summary['Cluster Size'] = df['Cluster'].value_counts().sort_index()

# Display the summary table, here we can analyze the key standout features for each archetype
print(cluster_summary.round(2))
```

**Output:**
```
        PTS     AST     REB    STL    BLK  FG_PCT  Cluster Size
0    119.65   26.30   55.79   9.94   5.93    0.41          2792
1   1274.22  362.94  353.77  87.04  28.88    0.46           768
2    923.07  128.44  583.52  53.34  91.13    0.54           546
3    594.84  121.85  239.66  45.07  24.16    0.46          2153
```

**Cluster Interpretations:**

Based on the cluster statistics, we can identify four distinct player archetypes:

- **Cluster 0 - Bench/Role Players (n=2,792):** The largest cluster with very low statistics across all categories (119.65 PTS, 26.30 AST, 55.79 REB). These are likely bench players or those with limited playing time who contribute minimally in all areas. Lower shooting efficiency (41%) suggests less skilled or less experienced players.

- **Cluster 1 - Elite All-Around Stars (n=768):** Players with exceptional statistics in all categories (1,274.22 PTS, 362.94 AST, 353.77 REB, 87.04 STL). These are likely MVP-caliber players and All-Stars who excel at scoring, playmaking, and contributing across the board. This is the smallest cluster of impact players, representing the league's elite.

- **Cluster 2 - Dominant Big Men/Centers (n=546):** High rebounds (583.52) and blocks (91.13) with excellent shooting efficiency (54%) but lower assists (128.44). These are likely traditional centers and power forwards who dominate in the paint, protect the rim, and score efficiently near the basket.

- **Cluster 3 - Solid Starters/Role Players (n=2,153):** Moderate statistics across all categories (594.84 PTS, 121.85 AST, 239.66 REB). These are likely consistent starters or high-level role players who contribute meaningfully but don't reach All-Star levels. They represent the league's reliable "glue guys."

**Key Insights:**
- The clustering successfully separated players into meaningful basketball archetypes based on their statistical profiles
- The largest groups are bench players (Cluster 0) and solid starters (Cluster 3), which aligns with typical NBA roster composition
- Elite players (Cluster 1) represent only about 12% of the dataset, which is consistent with the rarity of superstar talent
- Specialized big men (Cluster 2) form the smallest impact group, reflecting the evolution of positionless basketball in the modern NBA

---

## Feature Selection and Scaling

The dataset has already been reloaded and preprocessed in the previous step. The features `PTS`, `AST`, `REB`, `STL`, `BLK`, and `FG_PCT` were selected for clustering, standardized using `StandardScaler`, and then dimensionality reduction was applied using `PCA` with `n_components=4`. The resulting scaled and PCA-transformed data (`X_scaled` and `X_pca`) are ready for calculating the Silhouette Score.

### Calculate Silhouette Score

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Apply K-Means Clustering (K=4) to create the 'Cluster' column
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])

print(f"The Silhouette Score for the K-Means clustering (k=4) is: {silhouette_avg:.4f}")
```

**Output:**
```
The Silhouette Score for the K-Means clustering (k=4) is: 0.3103
```

**Interpretation:** The Silhouette Score of 0.3103 indicates a moderate level of separation and cohesion among the clusters. While this is not a perfect score (which would be close to 1.0), it suggests that the clusters are reasonably well-defined with some overlap. This score is acceptable for real-world sports data, where player roles often exist on a continuum rather than in discrete categories. The moderate score reflects the reality that some players exhibit characteristics of multiple archetypes.

---

## Visualize Clusters in PCA Space

Here we create a scatter plot using the first two principal components, with points colored according to their assigned K-Means cluster, to visually inspect the separation and distribution of the player archetypes.

```python
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['Cluster'],
    palette='viridis',
    legend='full'
)

# Map cluster numbers to archetype labels for the legend
archetype_map_for_legend = {str(k): v for k, v in archetype_map.items()}

handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [archetype_map_for_legend.get(label, label) for label in labels]
plt.legend(handles=handles, labels=new_labels, title='Player Archetype', loc='best')

plt.title('K-Means Clusters in PCA Space')
plt.xlabel('Overall Statistical Impact')
plt.ylabel('Contribution Focus (Higher values: Perimeter Play; Lower values: Interior Play)')
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Visualization:**

![K-Means Clusters in PCA Space](https://github.com/mahinkat/CMSC320_Final/blob/main/cmsc6.png?raw=true)

**Interpretation:** The scatter plot in PCA-reduced space reveals distinct groupings of the four player archetypes:
- The x-axis (PC1) represents overall statistical impact, with elite players to the right and limited-role players to the left
- The y-axis (PC2) captures the focus of contribution, with positive values indicating perimeter-oriented play (guards/playmakers) and negative values indicating interior play (big men)
- The visualization confirms reasonable cluster separation while also showing some natural overlap between adjacent archetypes, consistent with the moderate Silhouette Score

---

## Analyze Cluster Characteristics

We will examine the mean values of the original features within each cluster to provide a detailed understanding of what defines each player role.

```python
# Map the cluster index to archetype names for better readability
cluster_summary_labeled = cluster_summary.rename(index=archetype_map)
print(cluster_summary_labeled.round(2))
```

**Output:**
```
                                         PTS     AST     REB    STL    BLK  \
Limited Role/Fringe Player            119.65   26.30   55.79   9.94   5.93   
All-Around Guards/Playmakers         1274.22  362.94  353.77  87.04  28.88   
Defensive Bigs/Interior Specialists   923.07  128.44  583.52  53.34  91.13   
Secondary Role/Consistent Bench       594.84  121.85  239.66  45.07  24.16   

                                     FG_PCT  Cluster Size  
Limited Role/Fringe Player             0.41          2792  
All-Around Guards/Playmakers           0.46           768  
Defensive Bigs/Interior Specialists    0.54           546  
Secondary Role/Consistent Bench        0.46          2153  
```

---

## Data Analysis Summary

### Data Preparation
The `Regular_Season.csv` dataset was loaded, and unnecessary columns (`Unnamed: 0.1`, `Unnamed: 0`) were removed to ensure data cleanliness.

### Feature Engineering
Six key numerical features (`PTS`, `AST`, `REB`, `STL`, `BLK`, `FG_PCT`) were selected for clustering, standardized using `StandardScaler`, and then reduced to four principal components using PCA for efficient clustering.

### K-Means Clustering Performance
K-Means clustering with k=4 was applied, yielding a **Silhouette Score of 0.3103**, indicating a moderate level of separation and cohesion among the clusters. This score is reasonable for sports data where player roles often overlap.

### Cluster Visualizations
A scatter plot in the PCA-reduced space visually represented the four identified player archetypes, confirming their distribution and showing clear groupings with some expected overlap.

### Player Archetype Characteristics (Based on Mean Statistics)

#### Cluster 0: 'Limited Role/Fringe Player' (2,792 players - 44.6%)
- **Statistics:** 119.65 PTS, 26.30 AST, 55.79 REB, 9.94 STL, 5.93 BLK, 0.41 FG_PCT
- **Interpretation:** Characterized by very low average statistics across all categories. This group likely represents players with minimal on-court impact or limited playing time, including end-of-bench players, injured players with limited games, and rookies still developing.

#### Cluster 1: 'All-Around Guards/Playmakers' (768 players - 12.3%)
- **Statistics:** 1,274.22 PTS, 362.94 AST, 353.77 REB, 87.04 STL, 28.88 BLK, 0.46 FG_PCT
- **Interpretation:** Exhibits significantly higher averages in points, assists, rebounds, and steals, with a moderate field goal percentage. This cluster represents primary offensive contributors, franchise players, and well-rounded All-Stars who excel across multiple statistical categories.

#### Cluster 2: 'Defensive Bigs/Interior Specialists' (546 players - 8.7%)
- **Statistics:** 923.07 PTS, 128.44 AST, 583.52 REB, 53.34 STL, 91.13 BLK, 0.54 FG_PCT
- **Interpretation:** Defined by high average rebounds and blocks, alongside the **highest field goal percentage (54%)**. This cluster likely comprises centers and power forwards who excel at rebounding, shot-blocking, and scoring efficiently near the basket. Their low assist numbers reflect their interior-focused role.

#### Cluster 3: 'Secondary Role/Consistent Bench' (2,153 players - 34.4%)
- **Statistics:** 594.84 PTS, 121.85 AST, 239.66 REB, 45.07 STL, 24.16 BLK, 0.46 FG_PCT
- **Interpretation:** Shows moderate averages across statistics compared to Clusters 1 and 2. This group likely consists of quality role players, reliable bench contributors, and lower-tier starters who provide consistent production without specializing in any particular area. They are the "glue guys" essential for team depth.

### Key Findings
- The clustering successfully identified four meaningful and interpretable player archetypes
- Distribution is realistic: most players are role/bench players (79%), while elite players are rare (12.3%)
- Defensive specialists (8.7%) are the smallest group, reflecting modern basketball's emphasis on versatile players
- The highest shooting efficiency belongs to interior specialists, validating the advantage of close-range shots
- The moderate Silhouette Score reflects the natural continuum of player skills rather than perfectly discrete categories

---
