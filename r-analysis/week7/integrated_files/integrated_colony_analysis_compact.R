library(dplyr)
library(readr)

# Read the data
df <- read_csv("integrated_colony_analysis.csv")
df[df == ""] <- NA

# List of columns to collapse (all dyad-type columns)
dyad_cols <- c(
  "n_dyads_ff", "n_dyads_fm", "n_dyads_mf", "n_dyads_mm",
  "mean_percent_time_ff", "mean_percent_time_fm", "mean_percent_time_mf", "mean_percent_time_mm",
  "median_percent_time_ff", "median_percent_time_fm", "median_percent_time_mf", "median_percent_time_mm",
  "sd_percent_time_ff", "sd_percent_time_fm", "sd_percent_time_mf", "sd_percent_time_mm",
  "min_percent_time", "max_percent_time"
)

# List of network-level columns to keep (take first non-NA), EXCLUDING year and colony_clean
df_network_cols <- c(
  "Colony", "N_Birds", "N_Dyads", "Mean_SRI", "Edge_Density",
  "Avg_Clustering", "N_Communities", "Mean_Degree", "Mean_Betweenness", "Prop_Ambiguous"
)

# Collapse rows: for each group, take the first non-NA for each column
df_compact <- df %>%
  group_by(year, colony_clean) %>%
  summarise(across(all_of(c(dyad_cols, df_network_cols)), ~ first(na.omit(.))), .groups = "drop")

# Impute NA in n_dyads columns to 0
df_compact <- df_compact %>%
  mutate(
    n_dyads_ff = ifelse(is.na(n_dyads_ff), 0, n_dyads_ff),
    n_dyads_fm = ifelse(is.na(n_dyads_fm), 0, n_dyads_fm),
    n_dyads_mf = ifelse(is.na(n_dyads_mf), 0, n_dyads_mf),
    n_dyads_mm = ifelse(is.na(n_dyads_mm), 0, n_dyads_mm)
  )

# Remove rows where Colony is NA, empty, or only whitespace
df_compact <- df_compact %>%
  filter(!is.na(Colony), trimws(Colony) != "")

# Write to new CSV
write_csv(df_compact, "integrated_colony_analysis_compact.csv")

# ---
# To apply the same process to the integrated dyad analysis file, repeat the above steps with the appropriate file and columns. 