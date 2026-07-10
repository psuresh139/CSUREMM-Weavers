# supercolony_colony_summary_analysis.R
# Summarize centrality and dyad type distributions at colony and supercolony level
# Also compute supercolony size/diversity

# ---- Libraries ----
library(dplyr)
library(tidyr)
library(readr)
library(stringr)

# ---- File paths ----
colony_file <- 'integrated_files/integrated_colony_analysis.csv'
dyad_file <- 'dyad_full_integrated.csv'

# ---- Read data ----
colony_df <- read_csv(colony_file, show_col_types = FALSE)
print(colnames(colony_df))
dyad_df <- read_csv(dyad_file, show_col_types = FALSE)

# ---- 1.a. Centrality Comparison ----
# Colony-level
colony_centrality <- colony_df %>%
  dplyr::select(year, colony_clean, Mean_Degree, Mean_Betweenness) %>%
  group_by(colony_clean) %>%
  summarise(
    mean_degree = mean(Mean_Degree, na.rm=TRUE),
    median_degree = median(Mean_Degree, na.rm=TRUE),
    mean_betweenness = mean(Mean_Betweenness, na.rm=TRUE),
    median_betweenness = median(Mean_Betweenness, na.rm=TRUE),
    n_years = n()
  )
write_csv(colony_centrality, 'colony_centrality_summary.csv')

# Supercolony-level (if mapping available)
if('supercolony_id' %in% colnames(dyad_df)) {
  # Gather degree/betweenness for both birds
  bird_centrality <- dyad_df %>%
    dplyr::select(supercolony_id, bird1, bird2, degree_bird1, degree_bird2, betweenness_bird1, betweenness_bird2) %>%
    pivot_longer(cols = c(bird1, bird2), names_to = 'bird_role', values_to = 'bird') %>%
    mutate(
      degree = ifelse(bird_role == 'bird1', degree_bird1, degree_bird2),
      betweenness = ifelse(bird_role == 'bird1', betweenness_bird1, betweenness_bird2)
    ) %>%
    dplyr::select(supercolony_id, bird, degree, betweenness) %>%
    distinct()
  supercolony_centrality <- bird_centrality %>%
    group_by(supercolony_id) %>%
    summarise(
      mean_degree = mean(as.numeric(degree), na.rm=TRUE),
      median_degree = median(as.numeric(degree), na.rm=TRUE),
      mean_betweenness = mean(as.numeric(betweenness), na.rm=TRUE),
      median_betweenness = median(as.numeric(betweenness), na.rm=TRUE),
      n_birds = n_distinct(bird)
    )
  write_csv(supercolony_centrality, 'supercolony_centrality_summary.csv')
}

# ---- 1.b. Dyad Type Distribution ----
# Colony-level
dyad_types_colony <- dyad_df %>%
  group_by(colony, sex_pair) %>%
  summarise(n = n()) %>%
  group_by(colony) %>%
  mutate(prop = n / sum(n))
write_csv(dyad_types_colony, 'colony_dyad_type_distribution.csv')

# Supercolony-level
dyad_types_supercolony <- dyad_df %>%
  filter(!is.na(supercolony_id)) %>%
  group_by(supercolony_id, sex_pair) %>%
  summarise(n = n()) %>%
  group_by(supercolony_id) %>%
  mutate(prop = n / sum(n))
write_csv(dyad_types_supercolony, 'supercolony_dyad_type_distribution.csv')

# ---- 3. Supercolony Size and Diversity ----
supercolony_size <- dyad_df %>%
  filter(!is.na(supercolony_id)) %>%
  group_by(supercolony_id) %>%
  summarise(
    n_unique_birds = n_distinct(c(bird1, bird2)),
    n_unique_dyads = n_distinct(dyad),
    n_unique_colonies = n_distinct(colony),
    n_ff = sum(sex_pair == 'ff', na.rm=TRUE),
    n_fm = sum(sex_pair == 'fm', na.rm=TRUE),
    n_mf = sum(sex_pair == 'mf', na.rm=TRUE),
    n_mm = sum(sex_pair == 'mm', na.rm=TRUE)
  )
write_csv(supercolony_size, 'supercolony_size_diversity_summary.csv')

cat('Summary tables written: colony_centrality_summary.csv, supercolony_centrality_summary.csv, colony_dyad_type_distribution.csv, supercolony_dyad_type_distribution.csv, supercolony_size_diversity_summary.csv\n') 


##Survey Stats

# Load libraries
library(dplyr)
library(readr)

# Set file paths
centrality_file <- "final_analysis/colony_centrality_summary.csv"
size_file <- "final_analysis/colony_dyad_type_distribution.csv"
dyadtype_file <- "final_analysis/supercolony_dyad_type_distribution.csv"

# Read data
cent <- read_csv(centrality_file, show_col_types = FALSE)
size <- read_csv(size_file, show_col_types = FALSE)
dyadtype <- read_csv(dyadtype_file, show_col_types = FALSE)

cat("\n--- Top Supercolonies by Mean Degree ---\n")
print(cent %>% arrange(desc(mean_degree)) %>% select(supercolony_id, mean_degree, median_degree) %>% head(5))

cat("\n--- Top Supercolonies by Mean Betweenness ---\n")
print(cent %>% arrange(desc(mean_betweenness)) %>% select(supercolony_id, mean_betweenness, median_betweenness) %>% head(5))

cat("\n--- Largest Supercolonies (by n_unique_birds) ---\n")
print(size %>% arrange(desc(n_unique_birds)) %>% select(supercolony_id, n_unique_birds, n_unique_dyads, n_unique_colonies) %>% head(5))

cat("\n--- Dyad Type Diversity (FF, FM, MF, MM) in Largest Supercolony ---\n")
largest <- size %>% arrange(desc(n_unique_birds)) %>% slice(1) %>% pull(supercolony_id)
print(size %>% filter(supercolony_id == largest) %>% select(n_ff, n_fm, n_mf, n_mm))

cat("\n--- Dyad Type Proportions in Each Supercolony (Top 5 by FF proportion) ---\n")
ff_props <- dyadtype %>% filter(sex_pair == "ff") %>% arrange(desc(prop)) %>% select(supercolony_id, prop)
print(ff_props %>% head(5))

# ---- Survey Stats: Colony and Supercolony Summary ----

# Read summary files
centrality_file <- "final_analysis/colony_centrality_summary.csv"
colony_dyadtype_file <- "final_analysis/colony_dyad_type_distribution.csv"
supercolony_dyadtype_file <- "final_analysis/supercolony_dyad_type_distribution.csv"

cent <- readr::read_csv(centrality_file, show_col_types = FALSE)
coltype <- readr::read_csv(colony_dyadtype_file, show_col_types = FALSE)
supertype <- readr::read_csv(supercolony_dyadtype_file, show_col_types = FALSE)

cat('\n--- Colony Centrality Survey Stats ---\n')
cat('Mean of mean_degree:', mean(cent$mean_degree, na.rm=TRUE), '\n')
cat('Median of mean_degree:', median(cent$mean_degree, na.rm=TRUE), '\n')
cat('Range of mean_degree:', range(cent$mean_degree, na.rm=TRUE), '\n')
cat('Mean of mean_betweenness:', mean(cent$mean_betweenness, na.rm=TRUE), '\n')
cat('Median of mean_betweenness:', median(cent$mean_betweenness, na.rm=TRUE), '\n')
cat('Range of mean_betweenness:', range(cent$mean_betweenness, na.rm=TRUE), '\n')

cat('\nTop colonies by mean_degree:\n')
print(
  cent %>%
    arrange(desc(mean_degree)) %>%
    dplyr::select(colony_clean, mean_degree, median_degree)
)
cat('\nTop colonies by mean_betweenness:\n')
print(
  cent %>%
    arrange(desc(mean_betweenness)) %>%
    dplyr::select(colony_clean, mean_betweenness, median_betweenness)
)

cat('\n--- Colony Dyad Type Distribution Survey Stats ---\n')
print(
  coltype %>%
    group_by(colony) %>%
    summarise(most_common_type = sex_pair[which.max(prop)], max_prop = max(prop)) %>%
    arrange(desc(max_prop))
)

cat('\n--- Supercolony Dyad Type Distribution Survey Stats ---\n')
print(
  supertype %>%
    group_by(supercolony_id) %>%
    summarise(most_common_type = sex_pair[which.max(prop)], max_prop = max(prop)) %>%
    arrange(desc(max_prop))
)