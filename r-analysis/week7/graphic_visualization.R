library(tidyverse)
library(igraph)
library(tidygraph)
library(ggraph)
library(ggforce)
library(readxl)
library(dplyr)
library(tidyr)

setwd("~/Code/Birds")

# Load dyad data
dyads <- read_excel("supercolony_test_dyads.xlsx") # Adjust path if needed

# Set supercolony to visualize (easy to switch)
target_supercolony <- "SC_2013_1"  # Change this to visualize a different supercolony

# Filter to selected supercolony
dyads_sub <- dyads %>%
  filter(supercolony_id == target_supercolony)

# Prepare edges and nodes for igraph
tmp_edges <- dyads_sub %>%
  select(from = bird1, to = bird2, weight = percent_time, sex_pair, colony)

tmp_nodes <- dyads_sub %>%
  select(name = bird1, sex = bird1_sex, colony, supercolony_id) %>%
  bind_rows(dyads_sub %>%
              select(name = bird2, sex = bird2_sex, colony, supercolony_id)) %>%
  distinct(name, .keep_all = TRUE)

# Set dyad strength threshold (percent time together)
dyad_strength_threshold <- 50  # Change this value to adjust filtering

# Filter edges by dyad strength
filtered_edges <- tmp_edges %>% filter(weight > dyad_strength_threshold)

# Add intra/inter-colony label to edges
filtered_edges <- filtered_edges %>%
  mutate(
    from_colony = tmp_nodes$colony[match(from, tmp_nodes$name)],
    to_colony = tmp_nodes$colony[match(to, tmp_nodes$name)],
    edge_type = ifelse(from_colony == to_colony, "intra_colony", "inter_colony")
  )

# Print summary of edge types
edge_type_summary <- filtered_edges %>%
  group_by(edge_type) %>%
  summarise(
    mean_strength = mean(weight),
    n = n()
  )
print(edge_type_summary)

# (Optional) To filter or style by edge_type in the plot, use filtered_edges$edge_type
# Example: filter(filtered_edges, edge_type == "intra_colony") for only intra-colony edges

# Filter nodes to only those present in filtered edges
filtered_birds <- unique(c(filtered_edges$from, filtered_edges$to))
filtered_nodes <- tmp_nodes %>% filter(name %in% filtered_birds)

# Build the filtered graph
g <- tbl_graph(nodes = filtered_nodes, edges = filtered_edges, directed = FALSE)
g <- g %>% mutate(betweenness = centrality_betweenness())
g <- g %>% mutate(bridger = betweenness > quantile(betweenness, 0.90))

# Use Kamada-Kawai layout for better separation
layout_df <- create_layout(g, layout = "kk")

# Ensure a 'colony' column exists for plotting (if not already present)
if (!"colony" %in% colnames(layout_df)) layout_df$colony <- layout_df$colony.x

# Ensure a 'betweenness' column exists for plotting
if (!"betweenness" %in% colnames(layout_df)) layout_df$betweenness <- layout_df$betweenness.x

# Read intercolony distance data
intercolony_dist <- read_excel("data/Environment/intercolony_distance.xlsx")

# Pivot to wide distance matrix format
colony_dist_matrix <- intercolony_dist %>%
  tidyr::pivot_wider(names_from = colony2, values_from = dist_m)
rownames(colony_dist_matrix) <- colony_dist_matrix$colony1
colony_dist_matrix$colony1 <- NULL
colony_dist_matrix <- as.matrix(colony_dist_matrix)

# Run classical MDS (k = 2 dimensions)
mds_coords <- cmdscale(as.dist(colony_dist_matrix), k = 2) %>%
  as.data.frame()
mds_coords$colony <- rownames(mds_coords)
colnames(mds_coords)[1:2] <- c("x_colony", "y_colony")

# Standardize colony names: remove underscores to match MDS coords
layout_df$colony <- gsub("_", "", layout_df$colony)
mds_coords$colony <- gsub("_", "", mds_coords$colony)

# Join colony coordinates
layout_df <- layout_df %>%
  left_join(mds_coords, by = "colony")

# Diagnostic: Find birds missing colony coordinates
missing_coords <- layout_df %>% filter(is.na(x_colony) | is.na(y_colony))
cat("Number of birds missing colony coordinates after standardization:", nrow(missing_coords), "\n")
if (nrow(missing_coords) > 0) {
  print(missing_coords %>% select(name, colony))
}

# Remove birds with missing coordinates for plotting
layout_df <- layout_df %>% filter(!is.na(x_colony) & !is.na(y_colony))

# Arrange any number of colonies evenly around a circle
colony_names <- sort(unique(layout_df$colony))
n_colonies <- length(colony_names)
circle_angles <- seq(0, 2*pi, length.out = n_colonies + 1)[- (n_colonies + 1)]
triangle_radius <- 5
circle_centers <- tibble(
  colony = colony_names,
  x_center = triangle_radius * cos(circle_angles),
  y_center = triangle_radius * sin(circle_angles)
)

# Assign circle centers to each bird and jitter around center
set.seed(42)
layout_df <- layout_df %>%
  select(-starts_with("x_colony"), -starts_with("y_colony")) %>%  # Remove any old columns
  left_join(circle_centers, by = "colony") %>%
  group_by(colony) %>%
  mutate(
    angle = runif(n(), 0, 2*pi),
    radius = runif(n(), 1, 2),
    x = x_center + radius * cos(angle),
    y = y_center + radius * sin(angle)
  ) %>%
  ungroup()

# --- BIRD-LEVEL NETWORK WITH MDS-BASED LAYOUT ---

# 1. Prepare node data (birds)
# 1. Read bird-to-colony mapping from identification.xlsx
# Read and clean the mapping, filter to 2013 assignments only
bird_id_map <- read_excel("data/Index/identification.xlsx") %>%
  janitor::clean_names() %>%
  mutate(year = lubridate::year(date)) %>%
  filter(year == 2013) %>%
  group_by(combo, colony) %>%
  mutate(n_assign = n()) %>%
  ungroup()

# For each bird (Combo), pick the colony with the most assignments in 2013 (not necessarily >50%)
primary_colony_combo <- bird_id_map %>%
  group_by(combo) %>%
  arrange(desc(n_assign), date) %>%
  slice(1) %>%
  ungroup() %>%
  select(combo, colony)

# Standardize Combo and colony names
standardize_colony <- function(x) toupper(gsub("_", "", x))
dyads_sub$bird1 <- toupper(trimws(dyads_sub$bird1))
dyads_sub$bird2 <- toupper(trimws(dyads_sub$bird2))
primary_colony_combo$combo <- toupper(trimws(primary_colony_combo$combo))
primary_colony_combo$colony <- standardize_colony(primary_colony_combo$colony)
dyads_sub$colony <- standardize_colony(dyads_sub$colony)
mds_coords$colony <- standardize_colony(mds_coords$colony)

# Build node and edge data using Combo
bird_nodes <- dyads_sub %>%
  select(name = bird1) %>%
  bind_rows(dyads_sub %>% select(name = bird2)) %>%
  distinct(name) %>%
  left_join(primary_colony_combo, by = c("name" = "combo"))

bird_edges <- dyads_sub %>%
  select(from = bird1, to = bird2, weight = percent_time, sex_pair) %>%
  left_join(primary_colony_combo, by = c("from" = "combo")) %>%
  rename(colony_from = colony) %>%
  left_join(primary_colony_combo, by = c("to" = "combo")) %>%
  rename(colony_to = colony) %>%
  mutate(
    edge_type = ifelse(colony_from == colony_to, "intra_colony", "inter_colony")
  )

# 0. Identify colonies in the current supercolony
supercolony_colonies <- unique(primary_colony_combo$colony[primary_colony_combo$colony %in% dyads_sub$colony])

# 1. Filter mds_coords to only include colonies in the current supercolony
mds_coords <- mds_coords %>%
  filter(colony %in% supercolony_colonies)

# 2. Filter bird_layout to only include birds in these colonies
bird_layout <- bird_nodes %>%
  filter(colony %in% supercolony_colonies)

# 3. Only keep edges where both endpoints are in these colonies
valid_birds <- bird_layout$name
bird_edges <- bird_edges %>%
  filter(from %in% valid_birds & to %in% valid_birds)

# 4. Proceed with MDS layout and plotting as before, using these colony assignments
# 4. Build the bird-level graph
# Remove x and y from node attributes before building the graph
bird_layout_graph <- bird_layout %>% select(name, colony)

g_bird <- tbl_graph(nodes = bird_layout_graph, edges = bird_edges, directed = FALSE)

# Plot with colony hulls and spatial separation
p <- ggraph(g_bird, layout = "manual", x = bird_layout$x, y = bird_layout$y) +
  geom_edge_link(
    aes(color = edge_type),
    width = 0.5,
    alpha = 0.2
  ) +
  scale_edge_color_manual(
    values = c(
      "intra_colony" = "gray70",
      "inter_colony" = "steelblue2"
    ),
    name = "Edge Type"
  ) +
  geom_mark_hull(
    data = bird_layout,
    aes(x = x, y = y, group = colony, fill = colony),
    concavity = 5, alpha = 0.10, show.legend = FALSE
  ) +
  geom_node_point(aes(color = colony), size = 3) +
  scale_color_brewer(palette = "Set2", na.value = "gray80") +
  scale_fill_brewer(palette = "Set2", na.value = "gray80") +
  theme_graph(base_family = "Helvetica") +
  labs(
    title = paste("Bird Network (Colony-Respecting Layout) for", target_supercolony),
    color = "Colony"
  ) +
  theme(legend.position = "right")


print(p)

