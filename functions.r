# List of functions used for Monte Carlo Simulation and Betting System for Premier League Predictions


# Function to calculate team parameters using Poisson regression
# Parameters(games) takes match data as input and returns a list containing team parameters
# and home advantage value.
Parameters <- function(games) {
  # Get unique team names and count
  teams <- sort(unique(c(games[,1], games[,2])), decreasing = FALSE)
  n <- length(teams)
  g <- nrow(games)
  
  # Initialize matrices to store response variables (Y) and predictor variables (X)
  Y <- matrix(0, 2*g, 1)
  X <- matrix(0, 2*g, ((2*n)+1))
  
  # Populate Y and X matrices based on match data
  for (i in 1:g) {
    Y[((2*i)-1)] <- games[i,3]
    Y[(2*i)] <- games[i,4]
    
    M <- which(teams == games[i,1])
    N <- which(teams == games[i,2])
    X[((2*i)-1),M] <- 1
    X[((2*i)-1),N+n] <- -1
    X[(2*i),N] <- 1
    X[(2*i),M+n] <- -1
    X[((2*i)-1),((2*n)+1)] <- 1
  }
  
  XX <- X[,-1]
  
  # Perform Poisson regression
  parameters <- glm(formula = Y ~ 0 + XX, family = poisson)
  Z <- c(0, coefficients(parameters))
  
  # Extract and organize parameter estimates
  P <- data.frame(row.names = teams, Attack = Z[1:n], Defence = Z[(n+1):(2*n)])
  
  # Extract home parameter estimate
  return(list(teams = P, home = as.numeric(Z[2*n+1])))
}

# Function to calculate team performance metrics based on match data
# Table(games) takes match data as input and returns a data frame containing team metrics.
Table <- function(games) {
  # Get unique team names and count
  teams <- sort(unique(c(games[,1], games[,2])), decreasing = FALSE)
  n <- length(teams)
  g <- nrow(games)
  
  # Initialize data frame to store team metrics
  T <- data.frame(Team=teams, P=matrix(0,n,1), HW=matrix(0,n,1), HD=matrix(0,n,1), HL=matrix(0,n,1),
                  HF=matrix(0,n,1), HA=matrix(0,n,1), AW=matrix(0,n,1), AD=matrix(0,n,1),
                  AL=matrix(0,n,1), AF=matrix(0,n,1), AA=matrix(0,n,1), GD=matrix(0,n,1),
                  Points=matrix(0,n,1))
  
  # Populate team metrics based on match outcomes
  for (i in 1:g) {
    home_team <- games[i,1]
    away_team <- games[i,2]
    home_goals <- games[i,3]
    away_goals <- games[i,4]
    
    # Update team metrics based on match outcome
    if (home_goals > away_goals) {
      T[which(teams == home_team),"Points"] <- T[which(teams == home_team),"Points"] + 3
      T[which(teams == home_team),"HW"] <- T[which(teams == home_team),"HW"] + 1
      T[which(teams == away_team),"AL"] <- T[which(teams == away_team),"AL"] + 1
    } else {
      if (home_goals == away_goals) {
        T[which(teams == home_team),"Points"] <- T[which(teams == home_team),"Points"] + 1
        T[which(teams == away_team),"Points"] <- T[which(teams == away_team),"Points"] + 1
        T[which(teams == home_team),"HD"] <- T[which(teams == home_team),"HD"] + 1
        T[which(teams == away_team),"AD"] <- T[which(teams == away_team),"AD"] + 1
      } else {
        T[which(teams == away_team),"Points"] <- T[which(teams == away_team),"Points"] + 3
        T[which(teams == away_team),"AW"] <- T[which(teams == away_team),"AW"] + 1
        T[which(teams == home_team),"HL"] <- T[which(teams == home_team),"HL"] + 1
      }
    }
    T[which(teams == home_team),"P"] <- T[which(teams == home_team),"P"] + 1
    T[which(teams == away_team),"P"] <- T[which(teams == away_team),"P"] + 1
    T[which(teams == home_team),"HF"] <- T[which(teams == home_team),"HF"] + home_goals
    T[which(teams == home_team),"HA"] <- T[which(teams == home_team),"HA"] + away_goals
    T[which(teams == away_team),"AF"] <- T[which(teams == away_team),"AF"] + away_goals
    T[which(teams == away_team),"AA"] <- T[which(teams == away_team),"AA"] + home_goals
    T[which(teams == home_team),"GD"] <- T[which(teams == home_team),"GD"] + (home_goals - away_goals)
    T[which(teams == away_team),"GD"] <- T[which(teams == away_team),"GD"] + (away_goals - home_goals)
  }
  
  # Create a summary table of team metrics ordered by Points and Goal Difference
  S <- data.frame(row.names = c(1:n), T[with(T, order(-Points, -GD)), ])
  return(S)
}

# Function to simulate match outcomes using Poisson distribution and team parameters
# Games(parameters, data_played) takes parameters and match data as input, and returns simulated match data.
Games <- function(parameters, data_played) {
  teams <- rownames(parameters$teams)
  n <- length(teams)
  C <- data.frame(HomeTeam = character(), AwayTeam = character(),
                  FTHG = numeric(), FTAG = numeric(),
                  stringsAsFactors = FALSE)
  
  # Create match combinations for teams that have not played yet
  for (i in 1:n) {
    for (j in 1:n) {
      if (i != j) {
        home_team <- teams[i]
        away_team <- teams[j]
        
        # Check if the combination is present in the data_played
        if (!any(data_played$HomeTeam == home_team & data_played$AwayTeam == away_team)) {
          C <- rbind(C, data.frame(HomeTeam = home_team, AwayTeam = away_team, FTHG = 0, FTAG = 0))
        }
      }
    }
  }
  
  # Simulate match outcomes using Poisson distribution and team parameters
  for (row in 1:nrow(C)) {
    hometeam <- C[row, "HomeTeam"]
    awayteam <- C[row, "AwayTeam"]
    
    a <- which(teams == hometeam)
    b <- which(teams == awayteam)
    lambdaa <- exp(parameters$teams[a, "Attack"] - parameters$teams[b, "Defence"] + parameters$home)
    lambdab <- exp(parameters$teams[b, "Attack"] - parameters$teams[a, "Defence"])
    
    # Simulate goals using Poisson distribution
    C[row, "FTHG"] <- rpois(1, lambdaa)
    C[row, "FTAG"] <- rpois(1, lambdab)
  }
  
  return(rbind(data_played, C))
}

# Function to simulate match outcomes and record results
# Sim(parameters, k) takes team parameters and number of simulations (k) as input,
# returns a data frame containing simulated match outcomes for each team.
Sim <- function(parameters, k) {
  teams <- rownames(parameters$teams)
  n <- length(teams)
  A <- data.frame(row.names=teams)
  
  # Simulate match outcomes and record results for k simulations
  for(i in 1:k) {
    T <- Table(Games(parameters))
    for(j in 1:n) {
      A[teams[j],i] <- which(T == teams[j])
    }
  }
  return(A)
}

# Function to calculate simulation statistics
# SimStats(Sim) takes simulation results as input and returns a data frame
# containing average, standard deviation, mode, attack, and defence values for each team.
SimStats <- function(Sim) {
  teams <- rownames(Sim)
  n <- length(teams)
  zero <- matrix(0,n,1)
  M <- data.frame(Team=teams, Average=rowMeans(Sim), StDev=zero, Mode=zero, Attack=zero,
                  Defence=zero)
  
  # Calculate simulation statistics for each team
  for(i in 1:n) {
    a <- as.numeric(Sim[i,])
    M[i,"StDev"] <- sd(a)
    M[i,"Mode"] <- names(sort(-table(a)))[1] 
  }
  
  # Copy attack and defence values from the original team parameters
  for(i in 1:n) {
    M[i,"Attack"] <- TeamParameters$teams[i,"Attack"]
    M[i,"Defence"] <- TeamParameters$teams[i,"Defence"]
  }
  
  N <- data.frame(row.names=c(1:n), M[with(M, order(Average)), ])
  return(N)
}

# Function to extract simulation results for a specific team
# SimTeam(Predictions, k, Team) takes simulation results, number of simulations (k),
# and a specific team as input, returns a vector of simulated match outcomes for that team.
SimTeam <- function(Predictions, k, Team) {
  D <- numeric(k)
  teams <- rownames(Predictions)
  
  # Extract simulated match outcomes for the specified team
  for(i in 1:k) {
    D[i] <- Predictions[which(teams == Team),i]
  }
  
  return(D)
}

# Function to simulate multiple parameter sets and calculate bias and standard deviation
# MultiPara(realpara, k) takes real team parameters and number of simulations (k) as input,
# returns a list containing bias and standard deviation values for Attack and Defence parameters.
MultiPara <- function(realpara, k) {
  teams <- rownames(realpara$teams)
  n <- length(teams)
  zero <- matrix(0,n,1)
  Q <- data.frame(row.names=teams, Attack=zero, ASQ=zero, Defence=zero, DSQ=zero)
  homepara <- 0
  homeSQ <- 0
  
  # Simulate games and calculate bias and standard deviation for Attack and Defence parameters
  for(i in 1:k) {
    G <- Games(realpara)
    P <- Parameters(G)
    for(j in 1:n) {
      Q[j,1] <- Q[j,1] + P$teams[j,1]
      Q[j,2] <- Q[j,2] + P$teams[j,1]^2
      Q[j,3] <- Q[j,3] + P$teams[j,2]
      Q[j,4] <- Q[j,4] + P$teams[j,2]^2	
    }
    homepara <- homepara + P$home
    homeSQ <- homeSQ + P$home^2
  }
  
  # Calculate bias and standard deviation for Attack, Defence, and Home parameters
  R <- data.frame(row.names=teams,
                  Attack.bias=realpara$teams[,1] - Q[,1]/k, 
                  Attack.sd=(k/(k-1))*(sqrt(Q[,2]/k - (Q[,1]/k)^2)),
                  Defence.bias=realpara$teams[,2] - Q[,3]/k, 
                  Defence.sd=(k/(k-1))*(sqrt(Q[,4]/k - (Q[,3]/k)^2)))
  
  return(list(teams=R, home.bias=realpara$home - homepara/k, 
              home.sd=(k/(k-1))*(sqrt(homeSQ/k - (homepara/k)^2))))
}

# Function to calculate probability table for two teams
# ProbTable(parameters, hometeam, awayteam) takes team parameters, home team, and away team as input,
# returns a probability table indicating match outcome probabilities.
ProbTable <- function(parameters, hometeam, awayteam) {
  teams <- rownames(parameters$teams)
  P <- parameters$teams
  home <- parameters$home
  a <- which(teams == hometeam)
  b <- which(teams == awayteam)
  
  # Calculate team-specific attack and defence factors
  lambdaa <- exp(P[a,]$Attack - P[b,]$Defence + home)
  lambdab <- exp(P[b,]$Attack - P[a,]$Defence)
  
  # Initialize probability vectors for different goal counts
  A <- numeric(8)
  B <- numeric(8)
  
  # Calculate match outcome probabilities for different goal counts
  for(i in 0:6) {
    A[(i+1)] <- dpois(i, lambdaa)
    B[(i+1)] <- dpois(i, lambdab)
  }
  
  # Calculate probability of 7 or more goals
  A[8] <- 1 - sum(A[1:7])
  B[8] <- 1 - sum(B[1:7])
  
  # Create column names for the probability table
  name <- c("0", "1", "2", "3", "4", "5", "6", "7+")
  
  # Initialize probability table with zeros
  zero <- numeric(8)
  C <- data.frame(row.names = name, "0" = zero, "1" = zero, "2" = zero, "3" = zero, "4" = zero,
                  "5" = zero, "6" = zero, "7+" = zero)
  
  # Populate the probability table with calculated probabilities
  for(j in 1:8) {
    for(k in 1:8) {
      C[j,k] <- A[k] * B[j]
    }
  }
  
  # Assign column names to the probability table
  colnames(C) <- name
  
  # Return the probability table with probabilities rounded to two decimal places
  return(round(C * 100, 2))
}

# Define a function to calculate Mean Absolute Error (MAE) for home and away goals
calculate_mae <- function(actual_home_goals, actual_away_goals, predicted_home_goals, predicted_away_goals) {
  # Calculate MAE for home goals
  mae_home <- mean(abs(predicted_home_goals - actual_home_goals))
  # Calculate MAE for away goals
  mae_away <- mean(abs(predicted_away_goals - actual_away_goals))
  
  return(list(mae_home = mae_home, mae_away = mae_away))
}

# Function to calculate result probabilities based on probability matrix
# ResultProbs(probs) takes a probability matrix as input and calculates probabilities of different match outcomes.
ResultProbs <- function(probs) {
  R <- matrix(0,3,1)
  n <- length(probs)
  
  # Calculate probabilities of different match outcomes
  for(i in 1:n) {
    for(j in 1:n) {
      if(i > j) {
        R[3] <- R[3] + probs[i,j]
      } else {
        if(i == j) {
          R[2] <- R[2] + probs[i,j]
        } else {
          R[1] <- R[1] + probs[i,j]
        }
      }
    }
  }
  return(list(HomeWin=R[1], Draw=R[2], AwayWin=R[3]))
}

# Function to simulate a betting scenario with noise
# Simulates a betting scenario by adding noise to odds and clipping them between 0 and 1.
# Input: Odds, number of bets.
# Output: Simulated bets with noise
simulate_bets <- function(odds, num_bets) {
  sigma <- 0.02
  bets <- rep(odds, num_bets)
  # Add random noise with a normal distribution to the odds
  noise <- rnorm(num_bets * length(odds), mean = 0, sd = sigma)
  bets <- pmin(pmax(bets + noise, 0), 1)  # Clip odds between 0 and 1
  return(bets)
}

# Function to calculate earnings for a specific match based on betting
# Calculates earnings for a specific match based on betting results and odds.
# Input: Match results, odds, number of bets.
# Output: House earnings from the match.
calculate_earnings_with_betting <- function(match_results, odds, num_bets) {
  bets <- simulate_bets(odds, num_bets)
  # Calculate total earnings from winning bets
  total_earnings <- sum(bets * match_results)
  # Calculate house earnings (total amount of bets placed - total earnings paid out)
  house_earnings <- num_bets - total_earnings
  return(house_earnings)
}

# Function to simulate a betting season and calculate total house earnings
# Simulates a betting season by calculating house earnings for each match and accumulating over the season.
# Input: Team parameters, match data, number of bets per match.
# Output: Total house earnings for the season.
simulate_betting_season <- function(parameters, data_played, num_bets_per_match) {
  teams <- rownames(parameters$teams)
  n <- length(teams)
  total_house_earnings <- 0
  
  for (i in 1:nrow(data_played)) {
    home_team <- data_played[i, "HomeTeam"]
    away_team <- data_played[i, "AwayTeam"]
    home_goals <- data_played[i, "FTHG"]
    away_goals <- data_played[i, "FTAG"]
    home_win_odds <- data_played[i, "HomeWin"]
    draw_odds <- data_played[i, "Draw"]
    away_win_odds <- data_played[i, "AwayWin"]
    
    a <- which(teams == home_team)
    b <- which(teams == away_team)
    lambdaa <- exp(parameters$teams[a, "Attack"] - parameters$teams[b, "Defence"] + parameters$home)
    lambdab <- exp(parameters$teams[b, "Attack"] - parameters$teams[a, "Defence"])
    
    # Simulate match result (0 for loss, 1 for win)
    match_results <- c(home_goals > away_goals, home_goals == away_goals, away_goals > home_goals)
    odds <- c(home_win_odds, draw_odds, away_win_odds)
    
    # Calculate house earnings for the current match and accumulate over the season
    match_earnings <- calculate_earnings_with_betting(match_results, odds, num_bets_per_match)
    total_house_earnings <- total_house_earnings + match_earnings
  }
  
  return(total_house_earnings)
}

