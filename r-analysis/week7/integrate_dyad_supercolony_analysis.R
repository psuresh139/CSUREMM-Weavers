# integrate_dyad_supercolony_analysis.R
# Script to integrate network metrics, dyad metadata, and supercolony data for analysis

library(tidyverse)
library(readxl)

# 1. Read and reshape network metric files
betweenness <- read_csv("betweenness_centrality_by_year.csv")
betweenness_long <- betweenness %>%
  pivot_longer(cols = starts_with("20") | starts_with("X20"), names_to = "year", values_to = "betweenness") %>%
  mutate(year = as.integer(str_replace(year, "^X", "")))

degree <- read_csv("degree_centrality_by_year.csv")
degree_long <- degree %>%
  pivot_longer(cols = starts_with("20") | starts_with("X20"), names_to = "year", values_to = "degree") %>%
  mutate(year = as.integer(str_replace(year, "^X", "")))

# 2. Read dyad strength (dyad-level)
dyad_strength <- read_csv("dyad_strength_by_year.csv") %>%
  pivot_longer(cols = starts_with("20") | starts_with("X20"), names_to = "year", values_to = "dyad_strength") %>%
  mutate(year = as.integer(str_replace(year, "^X", "")))

# 3. Read dyad metadata
dyad_meta <- read_excel("dyad_and_individual_percent_time_combined.xlsx") %>%
  mutate(year = as.integer(year))

# 4. Read supercolony data (adjust filename as needed)
supercolony <- read_excel("../dyads_with_supercolony.xlsx") %>%
  mutate(year = as.integer(year))

# 5. Join betweenness and degree for bird1
annotated_dyads <- dyad_meta %>%
  left_join(betweenness_long, by = c("bird1" = "individual", "year")) %>%
  rename(betweenness_bird1 = betweenness) %>%
  left_join(degree_long, by = c("bird1" = "individual", "year")) %>%
  rename(degree_bird1 = degree)

# 6. Join betweenness and degree for bird2
annotated_dyads <- annotated_dyads %>%
  left_join(betweenness_long, by = c("bird2" = "individual", "year")) %>%
  rename(betweenness_bird2 = betweenness) %>%
  left_join(degree_long, by = c("bird2" = "individual", "year")) %>%
  rename(degree_bird2 = degree)

# 6.1. Create dyad column (alphabetical order for consistency)
annotated_dyads <- annotated_dyads %>%
  mutate(dyad = paste(pmin(bird1, bird2), pmax(bird1, bird2), sep = "_"))

# 7. Join dyad strength (by dyad and year)
annotated_dyads <- annotated_dyads %>%
  left_join(dyad_strength, by = c("dyad", "year"))

# 8. Join supercolony data (by bird1, bird2, year, colony)
annotated_dyads <- annotated_dyads %>%
  left_join(supercolony, by = c("bird1", "bird2", "year", "colony"))

# 9. Clean up and create analysis columns
annotated_dyads <- annotated_dyads %>%
  mutate(
    sex_pair = coalesce(sex_pair.x, sex_pair.y),
    in_supercolony = dyad_in_supercolony
  )

# 10. Save the integrated dataset for further analysis
write_csv(annotated_dyads, "dyad_full_integrated.csv")

# Print a summary
glimpse(annotated_dyads)
summary(annotated_dyads) 