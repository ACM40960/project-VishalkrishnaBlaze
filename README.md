## Monte Carlo Simulation and Betting System for Premier League Predictions

![Rstudio](https://img.shields.io/badge/Rstudio-v4.2.1+-blue.svg)
![Python](https://img.shields.io/badge/python-v3.11.4+-red.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

Welcome to the **SportsStatPredict** project repository! This project aims to leverage machine learning algorithms to predict sports outcomes, project league standings, and provide insights into betting odds. By combining historical sports statistics and real-time odds data, this project enhances the understanding of sports events and their potential outcomes.

### Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Betting Insights](#betting-insights)
- [Contributing](#contributing)
- [License](#license)

### Project Overview

The **SportsStatPredict** project is designed to provide users with accurate predictions for sports outcomes, league standings, and betting odds comparisons. By utilizing machine learning techniques, historical sports statistics, and real-time odds data, the project aims to enhance sports enthusiasts' understanding of upcoming events.

### Data Sources

- Historical sports statistics: Source the relevant historical data for the sports leagues of interest. In the case of the English premier league, the data has been sourced from https://www.football-data.co.uk/englandm.php
- Real-time odds data: Integrated with APIs or scraping tools to retrieve real-time odds data from betting platforms such as Bet365, Blue Square, Bet&Win etc. Data structure as been described here https://www.football-data.co.uk/notes.txt

### Features

- **EDA**: The dataset offers a comprehensive view of English Premier League matches for the 2022-2023 season. It includes match results, stats, and diverse betting odds. This resource is valuable for analyzing match outcomes, betting trends, and statistical modeling in sports analytics.
- **Outcome Predictions**: Utilize multiple trained machine learning models to predict the outcomes of upcoming sports matches and league standings.
- **Betting Insights**: Compare real-time betting odds with predicted outcomes, offering insights into potentially profitable betting opportunities.
- **Monetary Conversion**: Convert betting odds into monetary terms, allowing users to understand potential returns on their bets.

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/ACM40960/project-shubidiwoop.git   
   cd SportsStatPredict
   ```
2. Ensure you have Rstudio:
   Rstudio can be installed through (https://posit.co/products/open-source/rstudio/). The project was implemented on v4.2.1. 
2. Install the required libraries mentioned in the first code chunk through the Rstudio console:
   ```
   install.packages("package-name")
   ```

### Usage

1. Make sure you have the dataset in your working directory. The game details dataset for the English Premier League's 2022-2023 season can be found in the [data](https://github.com/ACM40960/project-shubidiwoop/blob/main/data/E0.csv) folder of this repository.

2.   Run the RMarkdown script using the knit button or the shortcut Ctrl+Shift+K:
   ```
   rmarkdown::render()
   ```

3. View the generated reports' projected league standings, outcome predictions, and betting insights.

4. Open the Python script file containing the EDA code using a text editor, IDE (e.g., PyCharm, Visual Studio Code), or Jupyter Notebook. Execute the EDA scripts by either:
    - Using your IDE's "Run" feature.
    - Running cells in Jupyter Notebook.
    - Using the terminal with the python command and the script's filename.
    ```
    python script_filename.py
    ```
Ensure Python and the required libraries mentioned in the first code chunk are installed.

### Hyperparameters Evaluation

The Poisson distribution is used in simulating football matches via MCMC due to its fit for modeling event counts i.e., goals scored. The derived Poisson regression model effectively captures this pattern. Goal distribution graph supports its relevance, confirming its role in football match simulation.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Poisson distribution" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/goal_poisson.svg">
</div>

- Train-test split point: For each `k`, the code trains the forecasting model using historical data and calculates the MAE for predicting future observations. The average Mean Absolute Error (MAE) is computed for home and away teams separately for different values of training set sizes. The code identifies the `k` value corresponding to the lowest average MAE through the plot, which can be considered as the optimal training set size for the forecasting task.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Poisson distribution" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/home_advantage.svg">
</div>
  
- Home Ground Advantage: The football data analysis shows home teams have an edge which may be due to factors like fan support and familiarity with their ground. This leads to an extra parameter (home ground advantage) in the Poisson regression model. The graph based on the 22-23 season confirms higher goal scoring for home teams. This parameter enhances realism, reducing differences between simulated and actual outcomes, measured by Mean Squared Error (MSE).

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Poisson distribution" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/home2.svg">
</div>

### Methodology

1. Optimal split value derived from MAE.
2. Simulate EPL table using various approaches: 
  - GLM and Poisson-based Monte Carlo: Utilize Poisson regression through R's 'glm' function to simulate standings, calculating attack, defense strengths, and home advantage. 
  - Manual team data reduction for strengths: Compute strength based on average goals, ignoring nuances like opponent quality. 
  - PCA for team strength and table prediction: PCA captures key statistics using two components, representing over 95% variability, and integrates home advantage. Determine home advantage through cross-validation
  - Factor Analysis gave inconclusive results due to inconducive data dimensions.
3. Procrustes Algorithm compares non-metric MDS values from models.
4. Choose best model and predict odds/probabilities.
5. Compare predicted odds with actual organization-provided odds.
6. Translate odds to monetary terms, ensuring the house profits.

- Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in machine learning to transform high-dimensional data into a lower-dimensional space while preserving as much variance as possible. In SportsStatPredict project, PCA has been used to extract meaningful features such as the team strengths, incorporating the home team advantage, from sports statistics data. This has in turn been used to simulate the league standings, and the odds for a team to win a game. It can be seen that the first two principal components reflect more than 95% variability in the data. 

```R
team_data <- as.data.frame(t(Table285[, c("HF", "HA", "AF", "AA")]))# Standardize the data before performing PCA
# Perform PCA
pca_result1 <- prcomp(scale(team_data), center = TRUE, scale. = TRUE)
```

- In addition to Principal Component Analysis (PCA), the SportsStatPredict project also employs Non-Metric Multidimensional Scaling (MDS) for visualizing teams in a 2D space while preserving their relative ranks. MDS is a technique that aims to represent high-dimensional data in a lower-dimensional space, often for visualization purposes. Non-Metric MDS is utilized to map team data into a 2D space, allowing for an intuitive visualization of team relationships. This technique retains the relative differences between teams while projecting them onto a 2D plane, providing insights into team clusters, similarities, and disparities.
  
```R
# Non-Metric MDS for 2D visualization
library(MASS)
loc = isoMDS(dist(SimTable_actual), k=2, eig=TRUE)
```

### Results

- The SportsStatPredict project utilizes Procrustes analysis to compare the results obtained from different models. Procrustes analysis is a technique that aligns two sets of data points to best match their structures. In the context of this project, Procrustes analysis is used to align the MDS representations of different models, enabling a quantitative comparison of their predictions and visualizations in the 2D space. The scatter plot below visualizes the results of a Procrustes analysis performed on two sets of Non-metric Multidimensional scaled data of the actual standing and pca standing table. The blue points correspond to the aligned data points from the "Actual" set, while the red dashed line indicates the 1:1 reference line, highlighting how well the alignment matches the original data.

```R
procrustes(loc$points, loc2$points)
```

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Poisson distribution" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/procustes.svg">
</div>

- Furthemore, MAE and MAPE has been employed to compare the different models. MAE calculates the average absolute difference between each team's position in the actual standings and the corresponding position in the simulated standings. This metric provides an overall measure of positional accuracy. Furthermore, MAPE calculates the average percentage difference between each team's position in the actual standings and the corresponding position in the simulated standings. This metric provides insights into the relative accuracy of positional predictions.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Poisson distribution" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/team_points_barplot1.svg">
</div>

- Based on the results table, PCA has been identified as the preferred method to pursue further investigation and development in odds and betting.

### Betting Insights

- Discuss how the betting insights are generated by comparing predicted outcomes with real-time odds.
  The Monte Carlo simulation approach is employed to calculate probabilities for different outcomes (HomeWin, Draw, AwayWin) for various match outcomes between teams, and set into a dataframe. This display acts as a preliminary assessment of the model's accuracy and is particularly useful for gaining insights into its strengths and areas of improvement at a glance.

- The dataframe is then merged with additional home, draw and away winning odds data obtained from betting organizations, like B365H and IW. All these odds are then scaled/standardized, to bring the predicted probabilities and real-time odds to a common scale for accurate correlation calculations. Then correlation coefficients are calculated to quantify the strength and direction of the linear relationship between the predicted probabilities and the real-time odds (by Bet365 and IW) for different outcomes (HomeWin, Draw, AwayWin). 

- Spearman's Rank Correlation Coefficient is useful to compare the two columns as their relationship follows a monotonic pattern, enabling assessment of non-linear connections and ordinal data comparisons. The presented results below indicate a positive correlation (0.45) between our predicted odds and the actual odds, suggesting a favorable alignment between our predictive model and the real-time odds.
  
- Ensuring house profits:

To achieve this, a simulation-based approach is used to analyze the potential earnings and outcomes of a betting strategy applied to football match results. A function is used that generates simulated betting outcomes based on given odds and a specified number of bets. It adds randomness from both normal and uniform distributions to the initial bets and calculates the resulting betting values and the total earnings, which are made to be exponential to the number of bets made. As seen from the plot below, the total house earnings increase as the number of bets made increases.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Poisson distribution" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/moneyplot2.svg">
</div>

### Future Prospects

The current simulation-based approach, inspired by SportsStatPredict, holds exciting prospects for future applications. It can be extended to different sports leagues, seasons, and even esports. Enhancements can involve integrating real-time data for a more realistic house earnings model and comparing a variety of machine learning models like Random Forests and Neural Networks for improved predictions. This expansion could greatly amplify the methodology's versatility, accuracy, and relevance for sports analytics and the betting industry.

## Conclusion

In conclusion, this project demonstrates a holistic approach that combines advanced statistical modeling, machine learning techniques, and real-time odds comparison to accurately predict sports outcomes, enabling informed betting decisions while ensuring the sustainability of house profits, which can be replicated across various sports leagues and seasons.

### Contributing

We welcome contributions to the **SportsStatPredict** project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-new-feature`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-new-feature`
5. Open a pull request detailing your changes.

### License

This project is licensed under the MIT License. Please have a look at the [LICENSE.md](https://github.com/ACM40960/project-shubidiwoop/blob/main/LICENSE) for more details.

### References

- James Adam Gardner, Modeling and Simulating Football Results, 2011.
- J. F. C. Kingman. Poisson Processes. Oxford University Press, 1993.
- Rue H, Salvesen O. 2000 Prediction and retrospective analysis of soccer matches in a league. J. R. Stat. Soc. Ser. D (Stat.) 49, 399-418.
- Source: UCD 2023 Spring Multivariate Analysis coursework
- Source: "English Premier League and Championship Soccer Data" by Football-Data.co.uk.

---

Feel free to reach out to the project maintainers for any questions or clarifications.

- [Shubham Sharma](https://github.com/shubidiwoop)
- [Vishalkrishna Bhosle](https://github.com/vishalkrishnablaze)

**Disclaimer:** This project is for educational and informational purposes only. Betting and gambling carry risks, and this project does not provide financial advice. Always gamble responsibly.

---
