# =============================================================================
# OVERLAY SOCIAL NETWORK ON SPATIAL COLONY LAYOUT
# Combines spatial colony positions with social association data
# =============================================================================

library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(readxl)

# Read normalized rotated coordinates
coords <- read.csv("colony_coords_all_years_rotated.csv", stringsAsFactors = FALSE)

# Read SRI data for a specific year (let's use 2016 as it has the most data)
sri_data <- read_csv("sri_data/SRI_2016.csv", show_col_types = FALSE)

# Function to extract colony prefix from bird ID
extract_colony_prefix <- function(bird_id) {
  prefix <- substr(bird_id, 1, 2)
  colony_map <- c(
    "BM" = "MSTO", "KM" = "LLOD", "OM" = "SPRA",
    "WM" = "MSTO", "RM" = "LLOD", "YM" = "SPRA",
    "PM" = "MSTO", "MB" = "LLOD", "MK" = "SPRA",
    "MO" = "MSTO"
  )
  result <- colony_map[prefix]
  return(ifelse(is.na(result), "Unknown", result))
}

# Add colony information to SRI data
sri_with_colony <- sri_data %>%
  mutate(
    colony_1 = sapply(bird_1, extract_colony_prefix),
    colony_2 = sapply(bird_2, extract_colony_prefix)
  )

# Get colony centroids (average position of all birds in each colony)
colony_centroids <- coords %>%
  mutate(colony_prefix = substr(colony, 1, 4)) %>%
  group_by(colony_prefix) %>%
  summarise(
    x = mean(x_rot_norm, na.rm = TRUE),
    y = mean(y_rot_norm, na.rm = TRUE),
    n_colonies = n(),
    .groups = "drop"
  )

# Create social network edges between colonies
colony_network <- sri_with_colony %>%
  group_by(colony_1, colony_2) %>%
  summarise(
    mean_sri = mean(SRI, na.rm = TRUE),
    n_dyads = n(),
    .groups = "drop"
  ) %>%
  filter(colony_1 != colony_2) %>%  # Remove self-connections
  left_join(colony_centroids, by = c("colony_1" = "colony_prefix")) %>%
  rename(x1 = x, y1 = y) %>%
  left_join(colony_centroids, by = c("colony_2" = "colony_prefix")) %>%
  rename(x2 = x, y2 = y) %>%
  filter(!is.na(x1) & !is.na(x2))

# Plot
p <- ggplot() +
  # Draw social network edges
  geom_segment(data = colony_network, 
               aes(x = x1, y = y1, xend = x2, yend = y2, 
                   color = mean_sri, size = n_dyads), 
               alpha = 0.7) +
  # Draw colony centroids
  geom_point(data = colony_centroids, aes(x = x, y = y), 
             size = 8, color = "black", shape = 21, fill = "white") +
  geom_text(data = colony_centroids, aes(x = x, y = y, label = colony_prefix), 
            size = 4, fontface = "bold") +
  # Color and size scales
  scale_color_viridis_c(option = "D", name = "Mean SRI", 
                       trans = "sqrt", breaks = c(0, 0.1, 0.2, 0.3)) +
  scale_size_continuous(name = "Number of Dyads", range = c(1, 5)) +
  labs(title = "Social Network Overlaid on Spatial Colony Layout (2016)",
       subtitle = "Edge thickness = number of dyads, Edge color = mean SRI",
       x = "Rotated X (normalized)", y = "Rotated Y (normalized)") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("spatial_social_network_overlay.png", p, width = 12, height = 10, dpi = 300)
cat("Saved spatial_social_network_overlay.png\n")

# Create CSV with colony coordinates
colony_coords_final <- coords %>%
  mutate(colony_prefix = substr(colony, 1, 4)) %>%
  group_by(colony_prefix) %>%
  summarise(
    x_norm = mean(x_rot_norm, na.rm = TRUE),
    y_norm = mean(y_rot_norm, na.rm = TRUE),
    n_colonies = n(),
    .groups = "drop"
  ) %>%
  arrange(colony_prefix)

write.csv(colony_coords_final, "colony_coordinates_final.csv", row.names = FALSE)
cat("Saved colony_coordinates_final.csv\n")

# Print summary
cat("\nColony coordinates summary:\n")
print(colony_coords_final)

cat("\nSocial network summary:\n")
print(colony_network %>% arrange(desc(mean_sri))) 