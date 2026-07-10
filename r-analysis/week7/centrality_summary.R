
# Load required libraries
library(igraph)
library(ggplot2)
library(dplyr)
library(viridis)
library(gridExtra)
library(tidyr)
library(readxl)
library(lubridate)
library(readr)

df <- read_csv("~/Code/Birds/week7/degree_centrality_by_year.csv")

head(df)
sum(df$`2013`, na.rm = TRUE)
sum(df$`2014`, na.rm = TRUE)
sum(df$`2015`, na.rm = TRUE)
sum(df$`2016`, na.rm = TRUE)

year_sums <- colSums(df[ , 1:4])
total_sum <- sum(year_sums)

normalized <- year_sums / total_sum

print(normalized)