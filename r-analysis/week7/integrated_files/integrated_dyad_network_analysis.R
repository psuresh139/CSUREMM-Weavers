# Multi-Level Dyad-Network Analysis
# This script implements a three-tiered approach to analyze relationships between 
# dyad percent time data and network characteristics:
# 
# Tier 1: Complete Network Analysis (Degree Centrality) - All dyads with degree data
# Tier 2: Core Network Analysis (Betweenness Centrality) - Only dyads in main component  
# Tier 3: Network Position Comparison - High vs Low degree birds
#
# This approach respects network theory and maximizes data usage appropriately

library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(tidyr)
library(viridis)
library(gridExtra)

setwd("~/Code/Birds/week7")

# Helper function for safe correlation calculation
safe_cor <- function(x, y) {
  # Remove NA values from both vectors
  complete_cases <- !is.na(x) & !is.na(y)
  x_clean <- x[complete_cases]
  y_clean <- y[complete_cases]
  
  # Check if we have enough data points
  if(length(x_clean) < 2) {
    return(NA_real_)
  }
  
  # Calculate correlation
  tryCatch({
    cor(x_clean, y_clean, use = "complete.obs")
  }, error = function(e) {
    NA_real_
  })
}

# Read the datasets
dyad_data <- read_excel("cleaned_dyad_percent_time_data.xlsx")
colony_network <- read.csv("colony_network_analysis.csv")
dyad_strength <- read.csv("network_data/dyad_strength_by_year.csv")
degree_centrality <- read.csv("network_data/degree_centrality_by_year.csv")
betweenness_centrality <- read.csv("network_data/betweenness_centrality_by_year.csv")

# Initial data quality check
cat("\n=== INITIAL DATA QUALITY CHECK ===\n")
cat("Dyad data dimensions:", dim(dyad_data), "\n")
cat("Colony network dimensions:", dim(colony_network), "\n")
cat("Dyad strength dimensions:", dim(dyad_strength), "\n")
cat("Degree centrality dimensions:", dim(degree_centrality), "\n")
cat("Betweenness centrality dimensions:", dim(betweenness_centrality), "\n")

# Clean and prepare dyad data
cat("\n=== CLEANING DYAD DATA ===\n")
dyad_data_clean <- dyad_data %>%
  filter(!is.na(percent_time) & !is.na(sex_pair)) %>%
  mutate(
    dyad_id = paste(bird1, bird2, sep = "_"),
    colony_clean = gsub("_", "", colony),
    bird1_sorted = pmin(bird1, bird2),
    bird2_sorted = pmax(bird1, bird2),
    dyad_id_network = paste(bird1_sorted, bird2_sorted, sep = "_")
  )

cat("After cleaning:", nrow(dyad_data_clean), "dyad records\n")
cat("Sex pair distribution:\n")
print(table(dyad_data_clean$sex_pair))

# Reshape network data
cat("\n=== RESHAPING NETWORK DATA ===\n")

# Dyad strength data
dyad_strength_long <- dyad_strength %>%
  pivot_longer(
    cols = starts_with("X"),
    names_to = "year",
    names_pattern = "X(\\d+)",
    values_to = "dyad_strength"
  ) %>%
  mutate(year = as.integer(year)) %>%
  filter(dyad_strength > 0)

# Degree centrality data
degree_long <- degree_centrality %>%
  pivot_longer(
    cols = starts_with("X"),
    names_to = "year",
    names_pattern = "X(\\d+)",
    values_to = "degree"
  ) %>%
  mutate(year = as.integer(year)) %>%
  filter(degree > 0)

# Betweenness centrality data
betweenness_long <- betweenness_centrality %>%
  pivot_longer(
    cols = starts_with("X"),
    names_to = "year",
    names_pattern = "X(\\d+)",
    values_to = "betweenness"
  ) %>%
  mutate(year = as.integer(year)) %>%
  filter(betweenness > 0)

cat("Network data summary:\n")
cat("Dyad strength records:", nrow(dyad_strength_long), "\n")
cat("Degree centrality records:", nrow(degree_long), "\n")
cat("Betweenness centrality records:", nrow(betweenness_long), "\n")

# Create integrated dyad-level dataset
cat("\n=== CREATING INTEGRATED DATASET ===\n")

dyad_level_analysis <- dyad_data_clean %>%
  # Join with dyad strength
  left_join(
    dyad_strength_long,
    by = c("dyad_id_network" = "dyad", "year" = "year")
  ) %>%
  # Join with degree centrality for both birds
  left_join(
    degree_long,
    by = c("bird1" = "individual", "year" = "year")
  ) %>%
  rename(degree_bird1 = degree) %>%
  left_join(
    degree_long,
    by = c("bird2" = "individual", "year" = "year")
  ) %>%
  rename(degree_bird2 = degree) %>%
  # Join with betweenness centrality for both birds
  left_join(
    betweenness_long,
    by = c("bird1" = "individual", "year" = "year")
  ) %>%
  rename(betweenness_bird1 = betweenness) %>%
  left_join(
    betweenness_long,
    by = c("bird2" = "individual", "year" = "year")
  ) %>%
  rename(betweenness_bird2 = betweenness) %>%
  # Calculate dyad-level metrics
  mutate(
    mean_degree = (degree_bird1 + degree_bird2) / 2,
    mean_betweenness = (betweenness_bird1 + betweenness_bird2) / 2,
    degree_difference = abs(degree_bird1 - degree_bird2),
    betweenness_difference = abs(betweenness_bird1 - betweenness_bird2),
    # Network position indicators
    has_degree_both = !is.na(degree_bird1) & !is.na(degree_bird2),
    has_betweenness_both = !is.na(betweenness_bird1) & !is.na(betweenness_bird2),
    has_dyad_strength = !is.na(dyad_strength),
    # Degree-based network position
    degree_position = case_when(
      mean_degree >= quantile(mean_degree, 0.75, na.rm = TRUE) ~ "High",
      mean_degree >= quantile(mean_degree, 0.25, na.rm = TRUE) ~ "Medium", 
      TRUE ~ "Low"
    )
  )

# Data coverage summary
cat("\n=== DATA COVERAGE SUMMARY ===\n")
coverage_summary <- dyad_level_analysis %>%
  summarise(
    total_dyads = n(),
    with_dyad_strength = sum(has_dyad_strength),
    with_degree_both = sum(has_degree_both),
    with_betweenness_both = sum(has_betweenness_both),
    pct_dyad_strength = round(with_dyad_strength / total_dyads * 100, 1),
    pct_degree_both = round(with_degree_both / total_dyads * 100, 1),
    pct_betweenness_both = round(with_betweenness_both / total_dyads * 100, 1)
  )
print(coverage_summary)

# ============================================================================
# TIER 1: COMPLETE NETWORK ANALYSIS (DEGREE CENTRALITY)
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("TIER 1: COMPLETE NETWORK ANALYSIS (DEGREE CENTRALITY)\n")
cat(strrep("=", 80), "\n")

# Filter for dyads with degree data
tier1_data <- dyad_level_analysis %>%
  filter(has_degree_both)

cat("\nTier 1 dataset:", nrow(tier1_data), "dyads with degree centrality\n")

# 1.1 Overall correlations
tier1_correlations <- tier1_data %>%
  summarise(
    cor_percent_degree = safe_cor(percent_time, mean_degree),
    cor_percent_strength = safe_cor(percent_time, dyad_strength),
    cor_degree_strength = safe_cor(mean_degree, dyad_strength)
  )

cat("\n1.1 Overall Correlations (Tier 1):\n")
print(tier1_correlations)

# 1.2 Sex pair analysis
tier1_sex_analysis <- tier1_data %>%
  group_by(sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    mean_degree = mean(mean_degree, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    # Safe correlation calculation with proper NA handling
    cor_percent_degree = safe_cor(percent_time, mean_degree),
    cor_percent_strength = safe_cor(percent_time, dyad_strength),
    .groups = 'drop'
  )

cat("\n1.2 Sex Pair Analysis (Tier 1):\n")
print(tier1_sex_analysis)

# 1.3 Year trends
tier1_year_trends <- tier1_data %>%
  group_by(year, sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    mean_degree = mean(mean_degree, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    .groups = 'drop'
  )

# 1.4 Visualizations
# Degree vs Percent Time
p1_tier1 <- ggplot(tier1_data, aes(x = percent_time, y = mean_degree, color = sex_pair)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~sex_pair, scales = "free") +
  labs(title = "Tier 1: Degree Centrality vs Percent Time by Sex Pair",
       x = "Percent Time Together", y = "Mean Degree Centrality",
       color = "Sex Pair") +
  theme_minimal() +
  scale_color_viridis_d()

# Year trends
p2_tier1 <- ggplot(tier1_year_trends, aes(x = year, y = mean_percent_time, color = sex_pair)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = "Tier 1: Mean Percent Time by Year and Sex Pair",
       x = "Year", y = "Mean Percent Time Together",
       color = "Sex Pair") +
  theme_minimal() +
  scale_color_viridis_d()

# ============================================================================
# TIER 2: CORE NETWORK ANALYSIS (BETWEENNESS CENTRALITY)
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("TIER 2: CORE NETWORK ANALYSIS (BETWEENNESS CENTRALITY)\n")
cat(strrep("=", 80), "\n")

# Filter for dyads with betweenness data (core network only)
tier2_data <- dyad_level_analysis %>%
  filter(has_betweenness_both)

cat("\nTier 2 dataset:", nrow(tier2_data), "dyads with betweenness centrality (core network)\n")

# 2.1 Overall correlations
tier2_correlations <- tier2_data %>%
  summarise(
    cor_percent_betweenness = safe_cor(percent_time, mean_betweenness),
    cor_percent_degree = safe_cor(percent_time, mean_degree),
    cor_percent_strength = safe_cor(percent_time, dyad_strength),
    cor_betweenness_degree = safe_cor(mean_betweenness, mean_degree)
  )

cat("\n2.1 Overall Correlations (Tier 2):\n")
print(tier2_correlations)

# 2.2 Sex pair analysis
tier2_sex_analysis <- tier2_data %>%
  group_by(sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    mean_degree = mean(mean_degree, na.rm = TRUE),
    mean_betweenness = mean(mean_betweenness, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    # Safe correlation calculation with proper NA handling
    cor_percent_betweenness = safe_cor(percent_time, mean_betweenness),
    cor_percent_degree = safe_cor(percent_time, mean_degree),
    cor_percent_strength = safe_cor(percent_time, dyad_strength),
    .groups = 'drop'
  )

cat("\n2.2 Sex Pair Analysis (Tier 2):\n")
print(tier2_sex_analysis)

# 2.3 Year trends
tier2_year_trends <- tier2_data %>%
  group_by(year, sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    mean_degree = mean(mean_degree, na.rm = TRUE),
    mean_betweenness = mean(mean_betweenness, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    .groups = 'drop'
  )

# 2.4 Visualizations
# Betweenness vs Percent Time
p1_tier2 <- ggplot(tier2_data, aes(x = percent_time, y = mean_betweenness, color = sex_pair)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~sex_pair, scales = "free") +
  labs(title = "Tier 2: Betweenness Centrality vs Percent Time by Sex Pair",
       x = "Percent Time Together", y = "Mean Betweenness Centrality",
       color = "Sex Pair") +
  theme_minimal() +
  scale_color_viridis_d()

# Degree vs Betweenness
p2_tier2 <- ggplot(tier2_data, aes(x = mean_degree, y = mean_betweenness, color = sex_pair)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~sex_pair, scales = "free") +
  labs(title = "Tier 2: Degree vs Betweenness Centrality by Sex Pair",
       x = "Mean Degree Centrality", y = "Mean Betweenness Centrality",
       color = "Sex Pair") +
  theme_minimal() +
  scale_color_viridis_d()

# ============================================================================
# TIER 3: NETWORK POSITION COMPARISON
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("TIER 3: NETWORK POSITION COMPARISON\n")
cat(strrep("=", 80), "\n")

# 3.1 High vs Low degree comparison
tier3_data <- dyad_level_analysis %>%
  filter(has_degree_both) %>%
  mutate(
    degree_category = case_when(
      mean_degree >= quantile(mean_degree, 0.75, na.rm = TRUE) ~ "High Degree",
      mean_degree <= quantile(mean_degree, 0.25, na.rm = TRUE) ~ "Low Degree",
      TRUE ~ "Medium Degree"
    )
  )

cat("\nTier 3 dataset:", nrow(tier3_data), "dyads categorized by degree position\n")

# 3.2 Network position analysis
tier3_position_analysis <- tier3_data %>%
  group_by(degree_category, sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    mean_degree = mean(mean_degree, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    has_betweenness_pct = mean(has_betweenness_both, na.rm = TRUE) * 100,
    .groups = 'drop'
  )

cat("\n3.1 Network Position Analysis:\n")
print(tier3_position_analysis)

# 3.3 Core vs Peripheral comparison
tier3_core_vs_peripheral <- dyad_level_analysis %>%
  filter(has_degree_both) %>%
  mutate(
    network_position = case_when(
      has_betweenness_both ~ "Core Network",
      TRUE ~ "Peripheral Network"
    )
  ) %>%
  group_by(network_position, sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_percent_time = mean(percent_time, na.rm = TRUE),
    mean_degree = mean(mean_degree, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    .groups = 'drop'
  )

cat("\n3.2 Core vs Peripheral Network Analysis:\n")
print(tier3_core_vs_peripheral)

# 3.4 Visualizations
# Network position comparison
p1_tier3 <- ggplot(tier3_data, aes(x = degree_category, y = percent_time, fill = sex_pair)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Tier 3: Percent Time by Network Position and Sex Pair",
       x = "Degree Category", y = "Percent Time Together",
       fill = "Sex Pair") +
  theme_minimal() +
  scale_fill_viridis_d()

# Core vs Peripheral
p2_tier3 <- ggplot(tier3_core_vs_peripheral, aes(x = network_position, y = mean_percent_time, fill = sex_pair)) +
  geom_col(position = "dodge", alpha = 0.8) +
  labs(title = "Tier 3: Mean Percent Time by Network Position",
       x = "Network Position", y = "Mean Percent Time Together",
       fill = "Sex Pair") +
  theme_minimal() +
  scale_fill_viridis_d()

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("COMPARATIVE ANALYSIS\n")
cat(strrep("=", 80), "\n")

# Compare results across tiers
comparative_summary <- bind_rows(
  # Tier 1 summary
  tier1_data %>%
    summarise(
      tier = "Tier 1 (Degree)",
      n_dyads = n(),
      mean_percent_time = mean(percent_time, na.rm = TRUE),
      mean_degree = mean(mean_degree, na.rm = TRUE),
      mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
      cor_percent_degree = safe_cor(percent_time, mean_degree),
      cor_percent_strength = safe_cor(percent_time, dyad_strength)
    ),
  # Tier 2 summary
  tier2_data %>%
    summarise(
      tier = "Tier 2 (Betweenness)",
      n_dyads = n(),
      mean_percent_time = mean(percent_time, na.rm = TRUE),
      mean_degree = mean(mean_degree, na.rm = TRUE),
      mean_betweenness = mean(mean_betweenness, na.rm = TRUE),
      mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
      cor_percent_betweenness = safe_cor(percent_time, mean_betweenness),
      cor_percent_degree = safe_cor(percent_time, mean_degree),
      cor_percent_strength = safe_cor(percent_time, dyad_strength)
    )
)

cat("\nComparative Summary Across Tiers:\n")
print(comparative_summary)

# ============================================================================
# SAVE RESULTS AND VISUALIZATIONS
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("SAVING RESULTS AND VISUALIZATIONS\n")
cat(strrep("=", 80), "\n")

# Save datasets
write.csv(tier1_data, "tier1_degree_analysis.csv", row.names = FALSE)
write.csv(tier2_data, "tier2_betweenness_analysis.csv", row.names = FALSE)
write.csv(tier3_data, "tier3_network_position_analysis.csv", row.names = FALSE)

# Save analysis results
write.csv(tier1_sex_analysis, "tier1_sex_analysis.csv", row.names = FALSE)
write.csv(tier2_sex_analysis, "tier2_sex_analysis.csv", row.names = FALSE)
write.csv(tier3_position_analysis, "tier3_position_analysis.csv", row.names = FALSE)
write.csv(comparative_summary, "comparative_analysis_summary.csv", row.names = FALSE)

# Save visualizations
pdf("tier1_degree_analysis.pdf", width = 12, height = 8)
print(p1_tier1)
dev.off()

pdf("tier1_year_trends.pdf", width = 10, height = 6)
print(p2_tier1)
dev.off()

pdf("tier2_betweenness_analysis.pdf", width = 12, height = 8)
print(p1_tier2)
dev.off()

pdf("tier2_degree_vs_betweenness.pdf", width = 12, height = 8)
print(p2_tier2)
dev.off()

pdf("tier3_network_position.pdf", width = 10, height = 6)
print(p1_tier3)
dev.off()

pdf("tier3_core_vs_peripheral.pdf", width = 10, height = 6)
print(p2_tier3)
dev.off()

# Create comprehensive summary report
cat("\n=== MULTI-LEVEL ANALYSIS SUMMARY ===\n")
cat("\nTier 1 (Degree Analysis):", nrow(tier1_data), "dyads analyzed\n")
cat("Tier 2 (Betweenness Analysis):", nrow(tier2_data), "dyads analyzed\n")
cat("Tier 3 (Network Position):", nrow(tier3_data), "dyads analyzed\n")

cat("\nKey Findings:\n")
cat("1. Degree centrality available for", coverage_summary$pct_degree_both, "% of dyads\n")
cat("2. Betweenness centrality available for", coverage_summary$pct_betweenness_both, "% of dyads\n")
cat("3. Dyad strength available for", coverage_summary$pct_dyad_strength, "% of dyads\n")

cat("\nFiles saved:\n")
cat("- tier1_degree_analysis.csv: Complete degree-based analysis\n")
cat("- tier2_betweenness_analysis.csv: Core network betweenness analysis\n")
cat("- tier3_network_position_analysis.csv: Network position comparisons\n")
cat("- Comparative analysis plots saved as PDF files\n")

cat("\nAnalysis complete!\n") 