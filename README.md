<p align="center">
  <img width="10%" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/football clipart.png">
</p>

<h1 align="center">
  <b>
    Monte Carlo Simulation and Betting System for Premier League Predictions
  </b>
</h1>

<p align="center">
  <img alt="Rstudio" src="https://img.shields.io/badge/Rstudio-v4.2.1+-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-v3.11.4+-red.svg">
  <img alt="Dependencies" src="https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg">
  <img alt="Contributions welcome" src="https://img.shields.io/badge/contributions-welcome-orange.svg">
</p>

Welcome to the project repository! This project aims to leverage machine learning algorithms to predict sports outcomes, league standings, and provide insights into betting odds. By combining historical sports statistics and real-time odds data, this project enhances the understanding of sports events and their potential outcomes.

<p align="center">
  <img src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/soccer-dribbble.gif">
</p>

**View this README in light mode for better graph visibility.**

### Table of Contents 📑
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters Evaluation](#hyperparameters-evaluation)
- [Project Workflow](#project-workflow)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Results](#results)
- [Betting Insights](#betting-insights)
- [Future Prospects](#future-prospects)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Data Sources](#data-sources)
- [Authors](#authors)
- [Poster](https://github.com/ACM40960/project-shubidiwoop/blob/main/Poster.png) <em> Click me for more details </em>

### Project Overview 

The project is designed to provide users with accurate predictions for sports outcomes, league standings, and betting odds comparisons. By utilizing machine learning techniques, historical sports statistics, and real-time odds data, the project aims to enhance sports enthusiasts' understanding of upcoming events. 
### Features

- **EDA**: The dataset offers a comprehensive view of English Premier League matches for the 2022-2023 season. It includes match results, stats, and diverse betting odds. This resource is valuable for analyzing match outcomes, betting trends, and statistical modeling in sports analytics.
- **Outcome Predictions**: Utilize multiple trained machine learning models to predict the outcomes of upcoming sports matches and league standings.
- **Betting Insights**: Compare real-time betting odds with predicted outcomes, offering insights into potentially profitable betting opportunities.
- **Monetary Conversion**: Convert betting odds into monetary terms, ensuring the betting house's profitability by analyzing potential returns on bets.

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/ACM40960/project-shubidiwoop.git   
   ```
2. RStudio is an integrated development environment (IDE) for R programming. It provides a user-friendly interface for writing and running R scripts, visualizing data, and generating reports. Ensure you have Rstudio: Rstudio can be installed using this [link](https://posit.co/products/open-source/rstudio/). The project was implemented on v4.2.1. 
3. Install the required libraries mentioned in the first code chunk of the rmarkdown through the Rstudio console:
   ```
   packages <- c("vegan", "ggplot2", "dplyr", "stats", "reshape2", "MASS")

   # Check if packages are not installed
   missing_packages <- setdiff(packages, installed.packages()[,"Package"])

   # Install missing packages
   if (length(missing_packages) > 0) {
     install.packages(missing_packages)
   } else {
     cat("All required packages are already installed.\n")
   }
   ```

### Usage

1. Make sure you have the dataset in your working directory. The game details dataset for the English Premier League's 2022-2023 season can be found in the [data](https://github.com/ACM40960/project-shubidiwoop/blob/main/data/E0.csv) folder of this repository. The historical sports statistics dataset used in this project has been sourced from [football-data.co.uk](https://www.football-data.co.uk/englandm.php), a valuable resource for football-related data. The website provides a wide range of historical data for previous seasons as well, allowing for comprehensive analysis and insights into football matches and outcomes.

2. Open the Python script [file](https://github.com/ACM40960/project-shubidiwoop/blob/main/EDA/EDA_Final_Project.ipynb) from the EDA folder containing the EDA code using a text editor, IDE (e.g., PyCharm, Visual Studio Code), or Jupyter Notebook. Execute the EDA scripts by either:
    - Using your IDE's "Run" feature.
    - Running cells in Jupyter Notebook.
    - Using the terminal with the python command and the script's filename.
    ```
    python script_filename.py
    ```
Ensure Python and the required libraries mentioned in the first code chunk are installed. If not, the following code can be used to install it locally. 
```
!pip install pandas seaborn matplotlib
```
The output of the EDA can be found in the EDA folder [here](https://github.com/ACM40960/project-shubidiwoop/blob/main/EDA/Final_project_EDA.html)

3. Run the RMarkdown [script](https://github.com/ACM40960/project-shubidiwoop/blob/main/project.Rmd) using the knit button or the shortcut Ctrl+Shift+K:
   ```
   rmarkdown::render()
   ```
   The output of the script as a pdf can be found [here](https://github.com/ACM40960/project-shubidiwoop/blob/main/project_r_output.pdf)

4. View the generated reports' projected league standings, outcome predictions, and betting insights.

### Hyperparameters Evaluation

The Poisson distribution simulates football matches via MCMC due to its fit for modeling event counts i.e., goals scored. The derived Poisson regression model effectively captures this pattern. The goal distribution graph supports its relevance, confirming its role in football match simulation.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="Poisson distribution" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/goal_poisson.svg">
   <p style="text-align: center;"><em>Figure 1: Graph illustrating the alignment of the Poisson distribution model with observed goal distribution in football matches.</em></p> 
</div>

- Train-test split point: For each `k`, the code trains the forecasting model using historical data and calculates the MAE for predicting future observations. The average Mean Absolute Error (MAE) is computed for home and away teams separately for different values of training set sizes. The code identifies the `k` value corresponding to the lowest average MAE through the plot, which can be considered as the optimal training set size for the forecasting task. Optimal k figured is around 228 (out of 380 matches played in a season)

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="k_ideal" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/average_mae_plot.svg">
   <p style="text-align: center;"><em>Figure 2: Scatter plot depicting the relationship between the parameter k and the average Mean Absolute Error (Avg_MAE= home and away average)</em></p> 
</div>
  
- Home Ground Advantage: The football data analysis shows home teams have an edge which may be due to factors like fan support and familiarity with their ground. This leads to an extra parameter (home ground advantage) in the Poisson regression model. The graph based on the 22-23 season confirms higher goal scoring for home teams. This parameter enhances realism, reducing differences between simulated and actual outcomes, measured by Mean Squared Error (MSE). This comes to around 0.45.

```R
  # Calculate lambdaa and lambdab with home advantage
     lambdaa <- exp(parameters$teams[a, "Attack"] - parameters$teams[b, "Defence"] + home_advantage)
     lambdab <- exp(parameters$teams[b, "Attack"] - parameters$teams[a, "Defence"])
```

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="home_advantage" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/home_advantage.svg">
   <p style="text-align: center;"><em>Figure 3: Barplot demonstrating the impact of home team advantage in football matches for the 22-23 season.</em></p>
</div>

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="home-advantage-mse" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/home2.svg">
   <p style="text-align: center;"><em>Figure 4: Relationship between Home Advantage and Mean Squared Error in football match simulation</em></p>
</div>

### Project Workflow

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

### Machine Learning Algorithms

- Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in machine learning to transform high-dimensional data into a lower-dimensional space while preserving as much variance as possible. In the project, PCA has been used to extract meaningful features such as the team strengths, incorporating the home team advantage, from sports statistics data. This has in turn been used to simulate the league standings, and the odds for a team to win a game. It can be seen that the first two principal components reflect more than 95% variability in the data. 

```R
team_data <- as.data.frame(t(Table285[, c("HF", "HA", "AF", "AA")]))# Standardize the data before performing PCA
# Perform PCA
pca_result <- prcomp(scale(team_data), center = TRUE, scale. = TRUE)
```

- Generalized Linear Models (GLMs) can play a significant role in this project by offering a versatile framework for predicting sports outcomes based on various factors. GLMs extend linear regression to accommodate non-normally distributed response variables, making them suitable for modeling binary outcomes like win/loss in sports. In the context of sports statistics, a GLM can be tailored to estimate the probabilities of different match outcomes by considering input features such as team strengths, home advantage, and previous performance. By applying appropriate link functions and distribution assumptions, GLMs can generate outcome probabilities and simulate league standings. Comparing the predicted standings with actual outcomes allows evaluation of the model's performance, aiding in the selection of the most effective predictive method.

```R
  parameters <- glm(formula = Y ~ 0 + XX, family = poisson)
# In the parameters function
```

- In addition, the project incorporates manual functions and the Poisson distribution to compute critical factors such as attack and defense strengths, along with the home advantage for each team. By analyzing historical data, these functions assess the average goals scored and conceded by each team. The attack strength is derived from the difference between average goals scored and conceded, while defense strength stems from the contrary difference. These calculated strengths form the basis for predicting match outcomes using the Poisson distribution. The λ (lambda) parameters, representing expected goal counts, are adjusted to include the home advantage, further enhancing the model's predictive accuracy.
  
```R
# Calculate Attack and Defence Strength for each team
team_data$Attack <- sapply(team_data$Team, function(team) {
  # Calculate average goals scored and conceded for the team
  avg_goals_scored <- mean(c(data_subset$FTHG[data_subset$HomeTeam == team], data_subset$FTAG[data_subset$AwayTeam == team]))
  avg_goals_conceded <- mean(c(data_subset$FTAG[data_subset$HomeTeam == team], data_subset$FTHG[data_subset$AwayTeam == team]))
  return(avg_goals_scored - avg_goals_conceded)
})
team_data$Defence <- sapply(team_data$Team, function(team) {
  # Calculate average goals scored and conceded for the team
  avg_goals_scored <- mean(c(data_subset$FTHG[data_subset$HomeTeam == team], data_subset$FTAG[data_subset$AwayTeam == team]))
  avg_goals_conceded <- mean(c(data_subset$FTAG[data_subset$HomeTeam == team], data_subset$FTHG[data_subset$AwayTeam == team]))
  return(avg_goals_conceded - avg_goals_scored)
})
```

- In addition to Principal Component Analysis (PCA), the project also employs Non-Metric Multidimensional Scaling (MDS) for visualizing teams in a 2D space while preserving their relative ranks. MDS is a technique that aims to represent high-dimensional data in a lower-dimensional space, often for visualization purposes. Non-Metric MDS is utilized to map team data into a 2D space, allowing for an intuitive visualization of team relationships. This technique retains the relative differences between teams while projecting them onto a 2D plane, providing insights into team clusters, similarities, and disparities.
  
```R
# Non-Metric MDS for 2D visualization
library(MASS)
loc = isoMDS(dist(SimTable_actual), k=2, eig=TRUE)
```

### Results

- The project utilizes Procrustes analysis to compare the results obtained from different models. Procrustes analysis is a technique that aligns two sets of data points to best match their structures. In the context of this project, Procrustes analysis is used to align the MDS representations of different models, enabling a quantitative comparison of their predictions and visualizations in the 2D space. The scatter plot below visualizes the results of a Procrustes analysis performed on two sets of Non-metric Multidimensional scaled data of the actual standing and pca standing table. The blue points correspond to the aligned data points from the "Actual" set, while the red dashed line indicates the 1:1 reference line, highlighting how well the alignment matches the original data.

```R
procrustes(loc$points, loc2$points)
```

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="procustes" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/procustes.svg">
   <p><em>Figure 5: Comparing PCA-derived points and actual league table points using Procrustes Analysis</em></p>
</div>

- Furthemore, MAE and MAPE has been employed to compare the different models. MAE calculates the average absolute difference between each team's position in the actual standings and the corresponding position in the simulated standings. This metric provides an overall measure of positional accuracy. Furthermore, MAPE calculates the average percentage difference between each team's position in the actual standings and the corresponding position in the simulated standings. This metric provides insights into the relative accuracy of positional predictions.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="standings" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/team_points_barplot1.svg">
   <p><em>Figure 6: Multiple barchart depicting comparison of team points using different prediction methods </em></p>
</div>

## Model Evaluation Metrics

| Metric          | GLM         | PCA         | Formula     |
|-----------------|-------------|-------------|-------------|
| MAE Score       | 1.9000000   | 3.8000000   | 2.7500000   |
| MAPE Score      | 4.5786319   | 8.0036573   | 5.4346673   |
| Procrustes Score| 3970.1353674| 3086.6280064| 5705.0707390|
| Correlation Score| 0.9992478  | 0.9992478   | 0.9984951   |


- Based on the results table, GLM has been identified as the preferred method to pursue further investigation and development in odds and betting.

### Betting Insights

- Discuss how the betting insights are generated by comparing predicted outcomes with real-time odds.
  The Monte Carlo simulation approach is employed to calculate probabilities for different outcomes (HomeWin, Draw, AwayWin) for various match outcomes between teams, and set into a dataframe. This display acts as a preliminary assessment of the model's accuracy and is particularly useful for gaining insights into its strengths and areas of improvement at a glance.

- The dataframe is then merged with additional home, draw and away winning odds data obtained from betting organizations, like B365H and IW. All these odds are then scaled/standardized, to bring the predicted probabilities and real-time odds to a common scale for accurate correlation calculations. Then correlation coefficients are calculated to quantify the strength and direction of the linear relationship between the predicted probabilities and the real-time odds (by Bet365 and IW) for different outcomes (HomeWin, Draw, AwayWin). 

- Spearman's Rank Correlation Coefficient is useful to compare the two columns as their relationship follows a monotonic pattern, enabling assessment of non-linear connections and ordinal data comparisons. The presented results below indicate a positive correlation (0.8) between our predicted odds and the actual odds, suggesting a favorable alignment between our predictive model and the real-time odds.


| No. | Comparison        | Correlation |
|----:|-------------------|------------:|
|   1 | Bet365 vs Homewin |     0.8109 |
|   2 | Bet365 vs Awaywin |     0.8465 |
|   3 | Average Bet365    |     0.8287 |
|   4 | IW vs Homewin     |     0.8032 |
|   5 | IW vs Awaywin     |     0.8463 |
|   6 | Average IW        |     0.8247 |
  
- Ensuring house profits:

To achieve this, a simulation-based approach is used to analyze the potential earnings and outcomes of a betting strategy applied to football match results. A function is used that generates simulated betting outcomes based on given odds and a specified number of bets. It adds randomness from both normal and uniform distributions to the initial bets and calculates the resulting betting values and the total earnings, which are made to be exponential to the number of bets made. As seen from the plot below, the total house earnings increase as the number of bets made increases.

<div style="background-color: white; display: inline-block; padding: 10px;">
    <img width="734" alt="moneyplot" src="https://github.com/ACM40960/project-shubidiwoop/blob/main/assets/moneyplot2.svg">
   <p><em>Figure 7: Relationship between the number of bets per match and total house earnings.</em></p>
</div>

### Future Prospects

The current simulation-based approach, inspired by the project, holds exciting prospects for future applications. It can be extended to different sports leagues, seasons, and even esports. Enhancements can involve integrating real-time data for a more realistic house earnings model and comparing a variety of machine learning models like Random Forests and Neural Networks for improved predictions. This expansion could greatly amplify the methodology's versatility, accuracy, and relevance for sports analytics and the betting industry.

### Conclusion

In conclusion, this project demonstrates a holistic approach that combines advanced statistical modeling, machine learning techniques, and real-time odds comparison to accurately predict sports outcomes, enabling informed betting decisions while ensuring the sustainability of house profits, which can be replicated across various sports leagues and seasons.

### Contributing

We welcome contributions to the project! To contribute:

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

### Data Sources

- Historical sports statistics: Source the relevant historical data for the sports leagues of interest. In the case of the English premier league, the data has been sourced [here](https://www.football-data.co.uk/englandm.php)
- Real-time odds data: Integrated with APIs or scraping tools to retrieve real-time odds data from betting platforms such as Bet365, Blue Square, Bet&Win etc. Data structure as been described [here](https://www.football-data.co.uk/notes.txt)

---
### Authors

Feel free to reach out to the project maintainers for any questions or clarifications.

- [Shubham Sharma](https://github.com/shubidiwoop)
- [Vishalkrishna Bhosle](https://github.com/vishalkrishnablaze)

**Disclaimer:** This project is for educational and informational purposes only. Betting and gambling carry risks, and this project does not provide financial advice. Always gamble responsibly.

---
