# =============================================================================
# CREATE SRI FILES FROM DYAD DATA
# Converts cleaned dyad data to SRI format for network analysis
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(purrr)

# Read the cleaned dyad data
cat("Reading dyad data...\n")
dyad_data <- read_csv("week7/collapsed_dyad_years_log.csv", show_col_types = FALSE)

# Check the structure
cat("Data structure:\n")
str(dyad_data)
cat("\nFirst few rows:\n")
print(head(dyad_data))

# Function to split dyad_id into bird_1 and bird_2
split_dyad_id <- function(dyad_id) {
  # Split by underscore and take first two parts
  parts <- str_split(dyad_id, "_")[[1]]
  if(length(parts) >= 2) {
    return(list(bird_1 = parts[1], bird_2 = parts[2]))
  } else {
    return(list(bird_1 = parts[1], bird_2 = parts[1]))
  }
}

# Process the data
cat("Processing dyad data...\n")
sri_data <- dyad_data %>%
  # Split dyad_id into individual birds
  mutate(
    dyad_parts = map(dyad_id, split_dyad_id),
    bird_1 = map_chr(dyad_parts, ~.x$bird_1),
    bird_2 = map_chr(dyad_parts, ~.x$bird_2)
  ) %>%
  # Select relevant columns and rename
  select(bird_1, bird_2, year, mean_association) %>%
  rename(SRI = mean_association) %>%
  # Remove rows where SRI is 0 (no association)
  filter(SRI > 0) %>%
  # Ensure bird_1 < bird_2 for consistency
  mutate(
    temp_bird_1 = pmin(bird_1, bird_2),
    temp_bird_2 = pmax(bird_1, bird_2),
    bird_1 = temp_bird_1,
    bird_2 = temp_bird_2
  ) %>%
  select(-temp_bird_1, -temp_bird_2)

# Check the processed data
cat("Processed data structure:\n")
str(sri_data)
cat("\nFirst few rows of processed data:\n")
print(head(sri_data))

# Get unique years
years <- unique(sri_data$year)
cat("\nYears found:", paste(years, collapse = ", "), "\n")

# Create SRI files for each year
cat("\nCreating SRI files for each year...\n")
for(year in years) {
  # Filter data for this year
  year_data <- sri_data %>%
    filter(year == !!year) %>%
    select(bird_1, bird_2, SRI)
  
  # Create filename
  filename <- paste0("week7/SRI_", year, ".csv")
  
  # Save to CSV
  write_csv(year_data, filename)
  
  cat("Created", filename, "with", nrow(year_data), "dyads\n")
  
  # Print summary statistics
  cat("  SRI range:", round(min(year_data$SRI), 4), "to", round(max(year_data$SRI), 4), "\n")
  cat("  Mean SRI:", round(mean(year_data$SRI), 4), "\n")
  cat("  Unique birds:", length(unique(c(year_data$bird_1, year_data$bird_2))), "\n\n")
}

# Create a summary file
summary_data <- sri_data %>%
  group_by(year) %>%
  summarise(
    n_dyads = n(),
    n_unique_birds = length(unique(c(bird_1, bird_2))),
    mean_sri = mean(SRI, na.rm = TRUE),
    min_sri = min(SRI, na.rm = TRUE),
    max_sri = max(SRI, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(summary_data, "week7/sri_summary_by_year.csv")
cat("Created week7/sri_summary_by_year.csv with summary statistics\n")

cat("\n=== SRI FILE CREATION COMPLETE ===\n")
cat("Files created:\n")
for(year in years) {
  cat("- week7/SRI_", year, ".csv\n", sep = "")
}
cat("- week7/sri_summary_by_year.csv\n")
cat("\nYou can now run networks_graphing.r!\n") 