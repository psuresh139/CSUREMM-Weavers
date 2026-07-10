# Dyad Percent Time Analysis by Sex Pair
# This script explores how dyadic co-presence and percent time differ by sex pair (MM, MF, FF)
# and connects these patterns to spatial and community characteristics.

library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)
setwd("~/Code/Birds")
# --- 1. Load the combined dyad/individual percent time data ---
combined_df <- readxl::read_excel("week7/dyad_and_individual_percent_time_combined.xlsx")

# --- 2. Standardize sex pair labels (MF and FM are the same) ---
combined_df <- combined_df %>%
  mutate(
    sex_pair_std = case_when(
      sex_pair %in% c("MF", "FM") ~ "MF",
      sex_pair == "MM" ~ "MM",
      sex_pair == "FF" ~ "FF",
      TRUE ~ sex_pair
    )
  )

# --- 3. Summary statistics by sex pair ---
cat("\nCounts of dyads by sex pair:\n")
print(table(combined_df$sex_pair_std))

cat("\nSummary statistics for co-presence and percent time by sex pair:\n")
print(combined_df %>%
  group_by(sex_pair_std) %>%
  summarise(
    n = n(),
    mean_copres = mean(co_presence_bins, na.rm = TRUE),
    median_copres = median(co_presence_bins, na.rm = TRUE),
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    median_percent_time = median(percent_time, na.rm = TRUE)
  ))

# --- 4. Visualize differences by sex pair ---
cat("\nGenerating boxplots for percent time and co-presence by sex pair...\n")
ggplot(combined_df, aes(x = sex_pair_std, y = percent_time)) +
  geom_boxplot() +
  labs(title = "Percent Time by Dyad Sex Pair", x = "Sex Pair", y = "Percent Time")
ggplot(combined_df, aes(x = sex_pair_std, y = co_presence_bins)) +
  geom_boxplot() +
  labs(title = "Co-presence Bins by Dyad Sex Pair", x = "Sex Pair", y = "Co-presence Bins")

# --- 5. Connect to spatial characteristics (example: colony) ---
cat("\nAverage co-presence by colony and sex pair:\n")
print(combined_df %>%
  group_by(colony, sex_pair_std) %>%
  summarise(mean_copres = mean(co_presence_bins, na.rm = TRUE)) %>%
  arrange(colony, sex_pair_std))

# --- 6. Statistical testing: ANOVA and Kruskal-Wallis ---
cat("\nANOVA: percent_time ~ sex_pair_std\n")
anova_result <- aov(percent_time ~ sex_pair_std, data = combined_df)
print(summary(anova_result))

cat("\nKruskal-Wallis: percent_time ~ sex_pair_std\n")
print(kruskal.test(percent_time ~ sex_pair_std, data = combined_df))

# --- 7. (Optional) Connect to community or social network data ---
# If you have a community or social metric column, join it in and repeat group_by analyses as above.
# Example:
# combined_df <- combined_df %>% left_join(community_df, by = c("bird1", "bird2", "year"))
# ... 

# --- 8. Integrate network features and analyze by sex pair ---
library(readxl)
library(dplyr)

# Load cleaned dyad percent time data (if not already loaded)
dyad_df <- readxl::read_excel("week7/cleaned_dyad_percent_time_data.xlsx")

# --- Reshape and join dyad strength ---
dyad_strength <- read.csv("week7/network_data/dyad_strength_by_year.csv")
dyad_strength_long <- dyad_strength %>%
  pivot_longer(cols = starts_with("X20"), names_to = "year", values_to = "strength") %>%
  mutate(
    year = as.numeric(gsub("X", "", year))  # remove "X" and convert to numeric
  ) %>%
  separate(dyad, into = c("bird1", "bird2"), sep = "_")

dyad_df <- dyad_df %>%
  left_join(dyad_strength_long, by = c("year", "bird1", "bird2"))

# --- Reshape and join degree centrality ---
degree_centrality <- read.csv("week7/network_data/degree_centrality_by_year.csv", check.names = FALSE)
degree_long <- degree_centrality %>%
  pivot_longer(cols = starts_with("20"), names_to = "year", values_to = "degree") %>%
  mutate(year = as.numeric(year)) %>%
  filter(!is.na(year))

dyad_df <- dyad_df %>%
  left_join(degree_long %>% rename(bird1_degree = degree), by = c("year", "bird1" = "individual")) %>%
  left_join(degree_long %>% rename(bird2_degree = degree), by = c("year", "bird2" = "individual"))

# --- Reshape and join betweenness centrality ---
betweenness_centrality <- read.csv("week7/network_data/betweenness_centrality_by_year.csv", check.names = FALSE)
betweenness_long <- betweenness_centrality %>%
  pivot_longer(cols = starts_with("20"), names_to = "year", values_to = "betweenness") %>%
  mutate(year = as.numeric(year)) %>%
  filter(!is.na(year))

dyad_df <- dyad_df %>%
  left_join(betweenness_long %>% rename(bird1_betweenness = betweenness), by = c("year", "bird1" = "individual")) %>%
  left_join(betweenness_long %>% rename(bird2_betweenness = betweenness), by = c("year", "bird2" = "individual"))

# --- Join community assignment for both birds (if available) ---
# (Assuming community file is already in long format with columns: year, id, community)
community <- read.csv("week7/colony_network_analysis.csv")
dyad_df <- dyad_df %>%
  left_join(community %>% rename(bird1_community = community), by = c("year", "bird1" = "id")) %>%
  left_join(community %>% rename(bird2_community = community), by = c("year", "bird2" = "id"))

dyad_df <- dyad_df %>%
  mutate(same_community = bird1_community == bird2_community)

cat("\nProportion of dyads in same community (all):\n")
print(mean(dyad_df$same_community, na.rm = TRUE))

cat("\nProportion of dyads in same community by sex pair:\n")
print(dyad_df %>%
  group_by(sex_pair) %>%
  summarise(prop_same_community = mean(same_community, na.rm = TRUE)))

cat("\nCorrelation between dyad strength and percent time (all):\n")
print(cor(dyad_df$strength, dyad_df$percent_time, use = 'complete.obs'))

cat("\nMean degree by sex pair:\n")
print(dyad_df %>%
  group_by(sex_pair) %>%
  summarise(mean_bird1_degree = mean(bird1_degree, na.rm = TRUE),
            mean_bird2_degree = mean(bird2_degree, na.rm = TRUE)))

# --- Expanded Analyses: Network Position and Time Spent Together ---
library(ggplot2)

# 1. Scatterplot: Dyad Strength vs Percent Time
cat("\nPlotting Dyad Strength vs Percent Time...\n")
ggplot(dyad_df, aes(x = strength, y = percent_time, color = sex_pair)) +
  geom_point(alpha = 0.5) +
  labs(title = "Dyad Strength vs. Percent Time", x = "Dyad Strength (SRI)", y = "Percent Time")

# 2. Colony-Level Aggregation
cat("\nColony-level aggregation of mean percent time and mean strength...\n")
colony_summary <- dyad_df %>%
  group_by(colony, year) %>%
  summarise(
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    mean_strength = mean(strength, na.rm = TRUE),
    n_dyads = n()
  ) %>%
  arrange(desc(mean_percent_time))
print(head(colony_summary, 10))

# 3. Linear Model: Predicting Percent Time
cat("\nLinear model: percent_time ~ strength + sex_pair + colony\n")
lm1 <- lm(percent_time ~ strength + sex_pair + colony, data = dyad_df)
print(summary(lm1))

# ---
# You can further expand analyses here:
# - Nest-level aggregation (if nest info is available)
# - Mixed models with random effects for year or colony
# - Visualization of network metrics by colony or nest
# - Subgroup analyses (e.g., by sex pair, year, or colony)
# ---

# --- Openings for further behavioral interpretation and subgroup analysis ---
# You can now filter dyad_df for FF, MF, MM and repeat any of the above analyses
# Example:
# ff_dyads <- dyad_df %>% filter(sex_pair == "FF")
# ...
# Add more analyses as needed for spatial, temporal, or social anchoring

# ... existing code ... 