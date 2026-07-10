# spatial_temporal_dyad_cleaning.R
# Purpose: Clean and merge spatial distance into dyad-year data for spatial-temporal modeling

library(dplyr)
library(readr)
library(readxl)
library(tidyr)
library(stringr)

setwd("~/Code/Birds")

# 1. Read in the temporal dyad data
raw <- read_csv("output/temporal_analysis/temporal_dyad_data.csv")

# 2. Select and rename key columns (add or remove as needed)
df <- raw %>%
  select(dyad_id, id1, id2, year, association, r, sex_combo, plot, V1sex, V2sex, uid1, uid2, pair_bond_strength, association_lag, total_disturbances, everything())

# 3. Map id1/id2 (combo) to kring using uid-ring-combo_index.xlsx (sheet 'unique')
combo_map <- readxl::read_excel("data/Index/uid-ring-combo_index.xlsx", sheet = "unique") %>%
  select(combo, metal) %>%
  mutate(combo = as.character(combo), metal = as.character(metal))

df <- df %>%
  left_join(combo_map, by = c("id1" = "combo")) %>%
  rename(kring1 = metal) %>%
  left_join(combo_map, by = c("id2" = "combo")) %>%
  rename(kring2 = metal)

# 4. Read log file and map kring1/kring2 to colony1/colony2 for each year
log <- read_excel("data/observation_logs/combined_weaver_log.xlsx") %>%
  mutate(year = lubridate::year(as.Date(Date)),
         colony = toupper(gsub("_", "", colony)),
         colony = ifelse(grepl("^[A-Z]+[0-9]$", colony), sub("([A-Z]+)([0-9])$", "\\10\\2", colony), colony),
         Kring = as.character(Kring))

id_colony <- log %>%
  arrange(Kring, year, desc(timestamp.1m)) %>%
  group_by(Kring, year) %>%
  slice(1) %>%
  ungroup() %>%
  select(Kring, year, colony)

df <- df %>%
  left_join(id_colony, by = c("kring1" = "Kring", "year" = "year")) %>%
  rename(colony1 = colony) %>%
  left_join(id_colony, by = c("kring2" = "Kring", "year" = "year")) %>%
  rename(colony2 = colony)

# 5. Standardize colony names again (just in case)
standardize_colony <- function(x) {
  x <- toupper(gsub("_", "", x))
  x <- ifelse(grepl("^[A-Z]+[0-9]$", x), sub("([A-Z]+)([0-9])$", "\\10\\2", x), x)
  x
}
df <- df %>% mutate(
  colony1 = standardize_colony(colony1),
  colony2 = standardize_colony(colony2)
)

# 6. Merge in intercolony distance
distmat <- read_excel("data/Environment/intercolony_distance.xlsx") %>%
  mutate(colony1 = standardize_colony(colony1),
         colony2 = standardize_colony(colony2))

df <- df %>%
  left_join(distmat, by = c("colony1", "colony2")) %>%
  mutate(dist_m = ifelse(is.na(dist_m),
                        distmat$dist_m[match(paste(colony2, colony1), paste(distmat$colony1, distmat$colony2))],
                        dist_m),
         dist_m = ifelse(colony1 == colony2, 0, dist_m))

# 7. Add pair_bond_lag (previous year's pair_bond_strength)
df <- df %>% arrange(dyad_id, year) %>% group_by(dyad_id) %>% mutate(pair_bond_lag = lag(pair_bond_strength, 1)) %>% ungroup()

# 8. Final cleaning: keep only relevant columns, remove duplicates/NAs
cleaned <- df %>%
  select(dyad_id, id1, id2, year, association, r, sex_combo, plot, colony1, colony2, dist_m, pair_bond_strength, pair_bond_lag, association_lag, total_disturbances) %>%
  distinct()

# Impute missing values
cleaned <- cleaned %>%
  mutate(
    total_disturbances = ifelse(is.na(total_disturbances), 0, total_disturbances),
    pair_bond_lag = ifelse(is.na(pair_bond_lag), 0, pair_bond_lag)
  )

# Identify dyad-years with multiple observations (observation-level duplicates)
dupes <- cleaned %>%
  group_by(dyad_id, year) %>%
  tally() %>%
  filter(n > 1)
cat("\nNumber of dyad-year combinations with >1 observation:", nrow(dupes), "\n")
if(nrow(dupes) > 0) {
  cat("Saving dyad-years with multiple observations to 'week6/duplicate_dyad_years.csv' for inspection.\n")
  dup_rows <- cleaned %>% semi_join(dupes, by = c("dyad_id", "year"))
  write_csv(dup_rows, "week6/duplicate_dyad_years.csv")
}

# Output all dyad-years (not just duplicates) for downstream SRI/network pipeline
all_dyad_years <- cleaned %>%
  group_by(dyad_id, id1, id2, year, sex_combo, plot, colony1, colony2, dist_m, pair_bond_strength, pair_bond_lag, association_lag, total_disturbances, r) %>%
  summarise(
    n_obs = n(),
    mean_association = mean(association, na.rm = TRUE),
    .groups = 'drop'
  )
write_csv(all_dyad_years, "week7/collapsed_dyad_years_log.csv")

cleaned <- cleaned %>%
  group_by(dyad_id, id1, id2, year, sex_combo, plot, colony1, colony2, dist_m, pair_bond_strength, pair_bond_lag, association_lag, total_disturbances, r) %>%
  summarise(association = mean(association, na.rm = TRUE), .groups = 'drop')

# Print preview of cleaned data
cat("\nPreview of cleaned dyad-year spatial data (after imputation and aggregation):\n")
print(head(cleaned, 10))
cat("\nColumn names:\n")
print(colnames(cleaned))

# Save cleaned data
write_csv(cleaned, "cleaned_dyad_year_spatial.csv")

cat("\nCleaned dyad-year spatial data saved as cleaned_dyad_year_spatial.csv in week6 folder.\n") 