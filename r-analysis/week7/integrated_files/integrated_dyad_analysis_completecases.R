library(readr)
library(dplyr)

# Read the data
df <- read_csv("integrated_dyad_analysis.csv")
df[df == ""] <- NA

# Keep only rows with no NA values in any column
df_complete <- df %>% filter(complete.cases(.))

# Write to new CSV
write_csv(df_complete, "integrated_dyad_analysis_completecases.csv")

# Print the number of rows remaining
cat("Number of complete rows:", nrow(df_complete), "\n") 