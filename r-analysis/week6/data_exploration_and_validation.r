# =============================================================================
# DATA EXPLORATION AND VALIDATION FOR WEAVER BIRD NETWORKS
# Cross-references network data with original dyad data
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(igraph)
library(viridis)
library(stringr)
library(purrr)

# =============================================================================
# 1. UNDERSTANDING NODES AND EDGES
# =============================================================================

cat("=== UNDERSTANDING NETWORK STRUCTURE ===\n\n")

# Read original dyad data
cat("Reading original dyad data...\n")
dyad_data <- read_csv("collapsed_dyad_years_log.csv", show_col_types = FALSE)

# Read SRI data for comparison
cat("Reading SRI data...\n")
sri_files <- list.files("sri_data", pattern = "SRI_.*\\.csv", full.names = TRUE)
years <- str_extract(basename(sri_files), "(?<=SRI_)[0-9]+(?=\\.csv)")

# Read summary data
summary_data <- read_csv("sri_data/sri_summary_by_year.csv", show_col_types = FALSE)

cat("\n=== WHAT ARE NODES AND EDGES? ===\n")
cat("NODES: Individual weaver birds (identified by their band codes)\n")
cat("  - Examples: BMOY, BMRY, BMWO, etc.\n")
cat("  - Each node represents one bird\n")
cat("  - Node names come from dyad_id (e.g., 'BMOY_BMRY' creates nodes 'BMOY' and 'BMRY')\n\n")

cat("EDGES: Social associations between bird pairs\n")
cat("  - Each edge connects two birds that have been observed together\n")
cat("  - Edge weight = SRI (Simple Ratio Index) = strength of association\n")
cat("  - SRI ranges from 0 (never together) to 1 (always together)\n")
cat("  - Only edges with SRI > 0 are included in networks\n\n")

# =============================================================================
# 2. DATA SOURCE VALIDATION
# =============================================================================

cat("=== DATA SOURCE VALIDATION ===\n\n")

# Check data flow
cat("Data Flow:\n")
cat("1. Original data: collapsed_dyad_years_log.csv\n")
cat("   - Contains: dyad_id, year, n_obs, mean_association\n")
cat("   - Total dyads:", nrow(dyad_data), "\n")
cat("   - Years:", paste(unique(dyad_data$year), collapse = ", "), "\n\n")

cat("2. Processed data: SRI_*.csv files\n")
cat("   - Created from: create_sri_files.r\n")
cat("   - Contains: bird_1, bird_2, SRI\n")
cat("   - Only includes dyads with mean_association > 0\n\n")

# Validate data transformation
cat("3. Validation of data transformation:\n")
for(year in years) {
  # Original data for this year
  orig_year <- dyad_data %>% 
    filter(year == as.numeric(year)) %>%
    filter(mean_association > 0)
  
  # SRI data for this year
  sri_file <- paste0("sri_data/SRI_", year, ".csv")
  sri_year <- read_csv(sri_file, show_col_types = FALSE)
  
  cat("   Year", year, ":\n")
  cat("     Original dyads with association > 0:", nrow(orig_year), "\n")
  cat("     SRI file dyads:", nrow(sri_year), "\n")
  cat("     Match:", nrow(orig_year) == nrow(sri_year), "\n")
}

# =============================================================================
# 3. NETWORK STATISTICS BY YEAR
# =============================================================================

cat("\n=== NETWORK STATISTICS BY YEAR ===\n\n")

# Create detailed summary
detailed_summary <- data.frame()

for(year in years) {
  sri_file <- paste0("sri_data/SRI_", year, ".csv")
  sri_data <- read_csv(sri_file, show_col_types = FALSE)
  
  # Get unique birds
  unique_birds <- unique(c(sri_data$bird_1, sri_data$bird_2))
  
  # Calculate network metrics
  g <- graph_from_data_frame(sri_data, directed = FALSE)
  E(g)$weight <- sri_data$SRI
  
  year_summary <- data.frame(
    Year = year,
    N_Dyads = nrow(sri_data),
    N_Birds = length(unique_birds),
    N_Edges = ecount(g),
    Mean_SRI = mean(sri_data$SRI),
    Median_SRI = median(sri_data$SRI),
    Min_SRI = min(sri_data$SRI),
    Max_SRI = max(sri_data$SRI),
    Edge_Density = edge_density(g),
    Avg_Clustering = transitivity(g, type = "global"),
    N_Communities = length(cluster_louvain(g, weights = E(g)$weight))
  )
  
  detailed_summary <- rbind(detailed_summary, year_summary)
}

print(detailed_summary)

# Save detailed summary
write_csv(detailed_summary, "network_detailed_summary.csv")
cat("\nSaved network_detailed_summary.csv\n")

# =============================================================================
# 4. BIRD IDENTIFICATION ANALYSIS
# =============================================================================

cat("\n=== BIRD IDENTIFICATION ANALYSIS ===\n\n")

# Collect all unique birds across all years
all_birds <- c()
for(year in years) {
  sri_file <- paste0("sri_data/SRI_", year, ".csv")
  sri_data <- read_csv(sri_file, show_col_types = FALSE)
  all_birds <- c(all_birds, sri_data$bird_1, sri_data$bird_2)
}
all_birds <- unique(all_birds)

cat("Total unique birds across all years:", length(all_birds), "\n")

# Analyze bird ID patterns
bird_patterns <- data.frame(
  bird_id = all_birds,
  prefix = substr(all_birds, 1, 2),
  length = nchar(all_birds)
)

pattern_summary <- bird_patterns %>%
  group_by(prefix, length) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count))

cat("\nBird ID patterns:\n")
print(pattern_summary)

# =============================================================================
# 5. CROSS-REFERENCE WITH ORIGINAL DATA
# =============================================================================

cat("\n=== CROSS-REFERENCE WITH ORIGINAL DATA ===\n\n")

# Function to split dyad_id
split_dyad_id <- function(dyad_id) {
  parts <- str_split(dyad_id, "_")[[1]]
  if(length(parts) >= 2) {
    return(list(bird_1 = parts[1], bird_2 = parts[2]))
  } else {
    return(list(bird_1 = parts[1], bird_2 = parts[1]))
  }
}

# Process original data to compare
orig_processed <- dyad_data %>%
  filter(mean_association > 0) %>%
  mutate(
    dyad_parts = map(dyad_id, split_dyad_id),
    bird_1 = map_chr(dyad_parts, ~.x$bird_1),
    bird_2 = map_chr(dyad_parts, ~.x$bird_2)
  ) %>%
  select(bird_1, bird_2, year, mean_association) %>%
  rename(SRI = mean_association)

# Compare with SRI data
cat("Cross-reference validation:\n")
for(year in years) {
  orig_year <- orig_processed %>% filter(year == as.numeric(year))
  sri_file <- paste0("sri_data/SRI_", year, ".csv")
  sri_year <- read_csv(sri_file, show_col_types = FALSE)
  
  # Check if all SRI dyads exist in original data
  sri_dyads <- paste(sri_year$bird_1, sri_year$bird_2, sep = "_")
  orig_dyads <- paste(orig_year$bird_1, orig_year$bird_2, sep = "_")
  
  missing_in_orig <- setdiff(sri_dyads, orig_dyads)
  missing_in_sri <- setdiff(orig_dyads, sri_dyads)
  
  cat("  Year", year, ":\n")
  cat("    Dyads in SRI but missing in original:", length(missing_in_orig), "\n")
  cat("    Dyads in original but missing in SRI:", length(missing_in_sri), "\n")
  if(length(missing_in_orig) > 0) {
    cat("    Missing dyads:", paste(head(missing_in_orig, 5), collapse = ", "), "\n")
  }
}

# =============================================================================
# 6. CREATE VALIDATION PLOTS
# =============================================================================

cat("\n=== CREATING VALIDATION PLOTS ===\n\n")

# Plot 1: Network size evolution
p1 <- ggplot(detailed_summary, aes(x = Year, y = N_Birds)) +
  geom_line(size = 1.2, color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "Number of Birds in Networks Over Time",
       x = "Year", y = "Number of Birds") +
  theme_minimal()

# Plot 2: SRI distribution
sri_all <- data.frame()
for(year in years) {
  sri_file <- paste0("sri_data/SRI_", year, ".csv")
  sri_data <- read_csv(sri_file, show_col_types = FALSE)
  sri_data$year <- year
  sri_all <- rbind(sri_all, sri_data)
}

p2 <- ggplot(sri_all, aes(x = SRI, fill = year)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  facet_wrap(~year, scales = "free_y") +
  labs(title = "SRI Distribution by Year",
       x = "SRI (Association Strength)", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# Plot 3: Network density evolution
p3 <- ggplot(detailed_summary, aes(x = Year, y = Edge_Density)) +
  geom_line(size = 1.2, color = "darkgreen") +
  geom_point(size = 3, color = "darkgreen") +
  labs(title = "Network Edge Density Over Time",
       x = "Year", y = "Edge Density") +
  theme_minimal()

# Save plots
ggsave("validation_network_size.png", p1, width = 8, height = 6, dpi = 300)
ggsave("validation_sri_distribution.png", p2, width = 12, height = 8, dpi = 300)
ggsave("validation_edge_density.png", p3, width = 8, height = 6, dpi = 300)

cat("Saved validation plots:\n")
cat("- validation_network_size.png\n")
cat("- validation_sri_distribution.png\n")
cat("- validation_edge_density.png\n")

# =============================================================================
# 7. SUMMARY REPORT
# =============================================================================

cat("\n=== SUMMARY REPORT ===\n\n")

cat("Data Sources:\n")
cat("- Original: collapsed_dyad_years_log.csv (", nrow(dyad_data), " dyad-year records)\n")
cat("- Processed: SRI_*.csv files in sri_data/ folder\n")
cat("- Years covered:", paste(years, collapse = ", "), "\n\n")

cat("Network Structure:\n")
cat("- Nodes: Individual weaver birds (identified by band codes)\n")
cat("- Edges: Social associations with SRI > 0\n")
cat("- Total unique birds across all years:", length(all_birds), "\n")
cat("- Total dyads across all years:", sum(detailed_summary$N_Dyads), "\n\n")

cat("Key Findings:\n")
cat("- Year with most birds:", detailed_summary$Year[which.max(detailed_summary$N_Birds)], "\n")
cat("- Year with most dyads:", detailed_summary$Year[which.max(detailed_summary$N_Dyads)], "\n")
cat("- Year with highest mean SRI:", detailed_summary$Year[which.max(detailed_summary$Mean_SRI)], "\n")
cat("- Year with highest edge density:", detailed_summary$Year[which.max(detailed_summary$Edge_Density)], "\n\n")

cat("Files Generated:\n")
cat("- network_detailed_summary.csv: Comprehensive network statistics\n")
cat("- validation_*.png: Data validation plots\n")
cat("- This script output: Detailed data exploration report\n\n")

cat("Next Steps:\n")
cat("1. Review the generated plots to understand network structure\n")
cat("2. Use network_detailed_summary.csv for statistical analysis\n")
cat("3. Cross-reference with your original research questions\n")
cat("4. Consider subsetting by bird characteristics (sex, age, etc.)\n")

cat("\n=== EXPLORATION COMPLETE ===\n") 