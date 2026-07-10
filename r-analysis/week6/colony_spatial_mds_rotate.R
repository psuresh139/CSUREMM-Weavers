# =============================================================================
# ROTATE NORMALIZED MDS COORDINATES TO MATCH REFERENCE ORIENTATION
# - Rotate about center (0.5, 0.5) by 40 degrees clockwise
# =============================================================================

library(dplyr)
library(ggplot2)

# Read the transformed coordinates
coords <- read.csv("colony_coords_all_years_transformed.csv", stringsAsFactors = FALSE)

# Center coordinates at (0,0)
coords$x_centered <- coords$x_norm - 0.5
coords$y_centered <- coords$y_norm - 0.5

# Rotation angle in radians (clockwise)
theta <- -40 * pi / 180  # -40 degrees

# Rotation matrix
rotate <- function(x, y, theta) {
  x_new <- x * cos(theta) - y * sin(theta)
  y_new <- x * sin(theta) + y * cos(theta)
  return(list(x = x_new, y = y_new))
}

rot <- rotate(coords$x_centered, coords$y_centered, theta)
coords$x_rot <- rot$x + 0.5
coords$y_rot <- rot$y + 0.5

# Optionally, re-normalize to [0,1] if any points are out of bounds
coords$x_rot_norm <- (coords$x_rot - min(coords$x_rot)) / (max(coords$x_rot) - min(coords$x_rot))
coords$y_rot_norm <- (coords$y_rot - min(coords$y_rot)) / (max(coords$y_rot) - min(coords$y_rot))

# Output rotated CSV
coords_out <- coords %>% select(colony, x_rot_norm, y_rot_norm)
write.csv(coords_out, "colony_coords_all_years_rotated.csv", row.names = FALSE)
cat("Saved colony_coords_all_years_rotated.csv\n")

# Plot for visual confirmation
p <- ggplot(coords_out, aes(x = x_rot_norm, y = y_rot_norm, label = colony)) +
  geom_point(size = 3, color = "purple") +
  geom_text(nudge_y = 0.02, size = 3) +
  labs(title = "Rotated MDS Spatial Layout of Colonies (40 deg CW)",
       x = "Rotated X", y = "Rotated Y") +
  theme_minimal()
ggsave("colony_mds_plot_all_years_rotated.png", p, width = 8, height = 6, dpi = 300)
cat("Saved colony_mds_plot_all_years_rotated.png\n")

cat("\nRotation complete!\n") 