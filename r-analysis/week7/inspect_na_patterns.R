# inspect_na_patterns.R
# Script to inspect NA patterns in annotated_dyads and suggest strategies

library(tidyverse)

# Read the integrated dataset
df <- read_csv("dyad_full_integrated.csv")

# Count NAs per column
na_counts <- sapply(df, function(x) sum(is.na(x)))
na_props <- na_counts / nrow(df)

# Create a summary table
na_summary <- tibble(
  column = names(na_counts),
  n_na = na_counts,
  prop_na = round(na_props, 3)
) %>%
  arrange(desc(prop_na))

print(na_summary)

# Print columns with >10% NAs
cat("\nColumns with >10% NAs:\n")
print(na_summary %>% filter(prop_na > 0.1))

# Suggest strategies
cat("\nSuggested strategies for handling missing data:\n")
cat("- For columns with very high NA proportion (>50%), consider dropping or using only for exploratory analysis.\n")
cat("- For columns with moderate NAs (10-50%), consider imputation, or filter to complete cases for analyses that require them.\n")
cat("- For columns with low NAs (<10%), you can filter out rows with NAs in those columns for analyses that require completeness.\n")
cat("- Always report how much data is used for each analysis, and consider if missingness is random or systematic.\n") 