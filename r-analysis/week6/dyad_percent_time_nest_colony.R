# Dyad Percent Time at Nest/Colony Script (Hourly Blocking, with ID mapping)
# Calculates, for each dyad, the percent of their total co-presence hours spent at each nest/colony/year
# Output is in the style of weaver_percent_time_nest_and_colony.xlsx, but for dyads

library(readxl)
library(dplyr)
library(purrr)
library(writexl)
library(lubridate)

setwd("~/Code/Birds")
# --- 1. Load data ---
obs_log <- read_excel("data/observation_logs/combined_weaver_log.xlsx")
dyads <- read.csv("data/behavior_social/Stable_Dyads_Over_Time.csv")
id_map <- read_excel("data/Index/identification.xlsx")

# --- 2. Prepare observation log ---
# Create a datetime column for easier matching
obs_log <- obs_log %>%
  mutate(
    datetime = as.POSIXct(paste(Date, Time), tz = "UTC"),
    Kring = as.character(Kring)
  )

# Extract year from Date if not present
obs_log <- obs_log %>%
  mutate(year = year(as.Date(Date)))

# --- 3. Map Kring to 4-letter combo code ---
# Assume id_map has columns: Metal (metal ring), combo (4-letter code)
# Adjust column names if needed
obs_log <- obs_log %>%
  left_join(id_map, by = c("Kring" = "Metal")) %>%
  mutate(bird_code = Combo)

# Add five minute block column
obs_log <- obs_log %>%
  mutate(five_min_block = floor_date(datetime, "5 minutes"))

# --- 4. Dyad co-presence calculation function (five minute blocks, using bird_code) ---
get_dyad_copresence_five_min <- function(b1, b2, obs_log) {
  log1 <- obs_log %>% filter(bird_code == b1)
  log2 <- obs_log %>% filter(bird_code == b2)
  
  # For each bird, get unique five_min/nest/colony/year combinations
  log1_blocks <- log1 %>% distinct(year, colony, nest, five_min_block)
  log2_blocks <- log2 %>% distinct(year, colony, nest, five_min_block)
  
  # Co-presence: both present in same nest/colony/5min/year
  merged <- inner_join(
    log1_blocks, log2_blocks,
    by = c("year", "colony", "nest", "five_min_block")
  )
  
  total_together <- nrow(merged)
  if (total_together == 0) return(NULL)
  
  summary <- merged %>%
    group_by(year, colony, nest) %>%
    summarise(
      co_presence_bins = n(),
      .groups = "drop"
    ) %>%
    mutate(
      percent_time = 100 * co_presence_bins / total_together,
      bird1 = b1,
      bird2 = b2
    )
  return(summary)
}

# --- 5. Apply to all dyads ---
dyad_results <- purrr::map2_dfr(
  dyads$id1, dyads$id2,
  ~get_dyad_copresence_five_min(.x, .y, obs_log)
)

# --- 6. Preview the results before exporting ---
print(head(dyad_results, 20))
str(dyad_results)
summary(dyad_results)

# --- 7. Output as CSV and Excel ---
write.csv(dyad_results, "dyad_percent_time_nest_colony.csv", row.names = FALSE)
write_xlsx(dyad_results, "dyad_percent_time_nest_colony.xlsx")

cat("Output saved to dyad_percent_time_nest_colony.csv and .xlsx in week6 folder\n") 


#### Looking for interesting feature

df <- read_excel("week6/dyad_percent_time_nest_colony.xlsx")

# --- 8. Combine dyad and individual percent times into a new summary file ---
# Load weaver data and id_map if not already loaded
weaver_df <- read_excel("weaver_percent_time_nest_and_colony.xlsx")
id_map <- read_excel("data/Index/identification.xlsx")

# Map Kring to Combo in weaver_df
weaver_df <- weaver_df %>%
  left_join(id_map, by = c("Kring" = "Metal")) %>%
  mutate(bird_code = Combo)

# For each dyad, get the individual percent times for both birds at each nest/colony/year
combined_df <- dyad_results %>%
  left_join(
    weaver_df %>% select(year, colony, nest, bird_code, bird1_percent_time = percent_time),
    by = c("year", "colony", "nest", "bird1" = "bird_code")
  ) %>%
  left_join(
    weaver_df %>% select(year, colony, nest, bird_code, bird2_percent_time = percent_time),
    by = c("year", "colony", "nest", "bird2" = "bird_code")
  ) %>%
  select(bird1, bird2, year, colony, nest, co_presence_bins, percent_time, bird1_percent_time, bird2_percent_time)

# Remove duplicate rows from combined_df
cat(sprintf("Rows in combined_df before deduplication: %d\n", nrow(combined_df)))
combined_df <- combined_df %>% distinct()
cat(sprintf("Rows in combined_df after deduplication: %d\n", nrow(combined_df)))

# Add sex combination column using weaver sex file
weaver_sex <- read_excel("data/observation_logs/weaver_sex.xlsx")

# Remove any existing sex columns to avoid duplicate names
combined_df <- combined_df %>%
  select(-any_of(c("bird1_sex", "bird2_sex", "sex_combo", "sex_pair")))

# Join sex info for both birds using 'combo'
combined_df <- combined_df %>%
  left_join(weaver_sex %>% select(combo, sex), by = c("bird1" = "combo")) %>%
  rename(bird1_sex = sex) %>%
  left_join(weaver_sex %>% select(combo, sex), by = c("bird2" = "combo")) %>%
  rename(bird2_sex = sex)

# Create the stitched sex_pair column (e.g., 'MF', 'FM', 'MM', 'FF', 'NN', etc.)
combined_df <- combined_df %>%
  mutate(
    sex_pair = paste0(
      ifelse(is.na(bird1_sex), "N", bird1_sex),
      ifelse(is.na(bird2_sex), "N", bird2_sex)
    )
  )

# Write to new Excel file
write_xlsx(combined_df, "dyad_and_individual_percent_time_combined.xlsx")

cat("Output saved to dyad_and_individual_percent_time_combined.xlsx in week6 folder\n")

# --- 9. Data exploration and interesting findings ---
cat("\n--- DATA EXPLORATION AND INTERESTING FINDINGS ---\n")

cat("\nPreview of combined_df table (first 10 rows):\n")
print(head(combined_df, 10))

# 1. Basic Distribution Summaries
cat("\nSummary statistics for dyad and individual percent times:\n")
print(summary(combined_df[, c("percent_time", "bird1_percent_time", "bird2_percent_time")]))

# 2. Correlation Analysis
combined_df$mean_individual <- rowMeans(combined_df[, c("bird1_percent_time", "bird2_percent_time")], na.rm = TRUE)
cor_val <- cor(combined_df$percent_time, combined_df$mean_individual, use = "complete.obs")
cat(sprintf("\nCorrelation between dyad percent time and mean individual percent time: %.3f\n", cor_val))

# 3. Difference Analysis
combined_df$diff_dyad_indiv <- combined_df$percent_time - combined_df$mean_individual
cat("\nSummary of difference (dyad percent time - mean individual percent time):\n")
print(summary(combined_df$diff_dyad_indiv))

# 4. Top/Bottom Dyads
cat("\nTop 5 dyads with highest co-presence:\n")
print(head(combined_df[order(-combined_df$co_presence_bins), ], 5))
cat("\nTop 5 dyads with lowest nonzero co-presence:\n")
print(head(combined_df[combined_df$co_presence_bins > 0, ][order(combined_df$co_presence_bins), ], 5))

# 5. Dyads with Large Discrepancies
cat("\nDyads where dyad percent time is much higher than mean individual percent time:\n")
print(head(combined_df[order(-combined_df$diff_dyad_indiv), ], 5))
cat("\nDyads where dyad percent time is much lower than mean individual percent time:\n")
print(head(combined_df[order(combined_df$diff_dyad_indiv), ], 5))

# 6. Presence Patterns by Colony/Nest/Year
cat("\nAverage dyad co-presence by colony:\n")
print(combined_df %>% group_by(colony) %>% summarise(mean_copres = mean(co_presence_bins, na.rm = TRUE)) %>% arrange(-mean_copres))
cat("\nAverage dyad co-presence by year:\n")
print(combined_df %>% group_by(year) %>% summarise(mean_copres = mean(co_presence_bins, na.rm = TRUE)) %>% arrange(year))

# 7. Proportion of Dyads with High/Low Overlap
cat(sprintf("\nProportion of dyads with >50%% percent time: %.2f%%\n", 100 * mean(combined_df$percent_time > 50, na.rm = TRUE)))
cat(sprintf("Proportion of dyads with <10%% percent time: %.2f%%\n", 100 * mean(combined_df$percent_time < 10, na.rm = TRUE)))

# 8. Birds Frequently in Dyads
library(tidyr)
top_birds <- combined_df %>%
  filter(co_presence_bins > quantile(co_presence_bins, 0.95, na.rm = TRUE)) %>%
  pivot_longer(cols = c(bird1, bird2), names_to = "bird_role", values_to = "bird") %>%
  group_by(bird) %>%
  summarise(count = n()) %>%
  arrange(-count)
cat("\nBirds most frequently in top 5% co-presence dyads:\n")
print(head(top_birds, 10))


df <- read_excel("week6/dyad_and_individual_percent_time_combined.xlsx")

df_unique <- df[!duplicated(df), ]

head(df_unique)
library(writexl)
write_xlsx(df_unique, "week6/cleaned_dyad_percent_time_data.xlsx")



# Print the result
print(df_unique)