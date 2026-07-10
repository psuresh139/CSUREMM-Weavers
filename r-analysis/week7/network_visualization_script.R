# ============================================================================
# NETWORK VISUALIZATION SCRIPT: WEAVER BIRD SOCIAL NETWORK
# ============================================================================

# Load required libraries
library(igraph)
library(ggplot2)
library(dplyr)
library(viridis)
library(gridExtra)
library(tidyr)
library(readxl)
library(lubridate)

cat("Loading network visualization libraries...\n")

# ============================================================================
# 1. LOAD AND PREPARE NETWORK DATA
# ============================================================================

# Load the network data
cat("Loading network data...\n")

# Read and reshape dyad strength data
cat("Reshaping dyad strength data...\n")
dyad_strength_data <- read.csv("network_data/dyad_strength_by_year.csv")
year_cols_ds <- grep("^X20[0-9]{2}$", colnames(dyad_strength_data), value = TRUE)
dyad_strength_long <- dyad_strength_data %>%
  tidyr::pivot_longer(
    cols = all_of(year_cols_ds),
    names_to = "year",
    values_to = "dyad_strength"
  ) %>%
  tidyr::separate(dyad, into = c("bird1", "bird2"), sep = "_")
dyad_strength_long$year <- as.integer(sub("^X", "", dyad_strength_long$year))

# Read and reshape degree centrality data
cat("Reshaping degree centrality data...\n")
degree_data <- read.csv("network_data/degree_centrality_by_year.csv")
year_cols_deg <- grep("^X20[0-9]{2}$", colnames(degree_data), value = TRUE)
degree_long <- degree_data %>%
  tidyr::pivot_longer(
    cols = all_of(year_cols_deg),
    names_to = "year",
    values_to = "mean_degree"
  )
degree_long$year <- as.integer(sub("^X", "", degree_long$year))

# Read and reshape betweenness centrality data
cat("Reshaping betweenness centrality data...\n")
betweenness_data <- read.csv("network_data/betweenness_centrality_by_year.csv")
year_cols_bet <- grep("^X20[0-9]{2}$", colnames(betweenness_data), value = TRUE)
betweenness_long <- betweenness_data %>%
  tidyr::pivot_longer(
    cols = all_of(year_cols_bet),
    names_to = "year",
    values_to = "mean_betweenness"
  )
betweenness_long$year <- as.integer(sub("^X", "", betweenness_long$year))

# Load dyad data for edge information
dyad_data <- read.csv("integrated_dyad_analysis.csv")

# 1. Read and filter identification data for 2017 and SPRA42
id_data <- readxl::read_excel("../data/index/identification.xlsx")
# Parse the year from the UTC Date column
id_data$Year <- lubridate::year(lubridate::ymd_hms(id_data$Date, quiet = TRUE))
# Filter for 2017 and colony SPRA42
id_2017_spra42 <- id_data %>%
  filter(Year == 2017, Colony == "SPRA42")
birds_2017_spra42 <- unique(id_2017_spra42$Combo)

# 2. Filter dyad data for 2017 and SPRA42 birds
year <- 2017
edges <- dyad_strength_long %>%
  filter(year == year,
         bird1 %in% birds_2017_spra42,
         bird2 %in% birds_2017_spra42) %>%
  select(bird1, bird2, dyad_strength) %>%
  filter(!is.na(dyad_strength) & dyad_strength > 0)

cat("Number of edges:", nrow(edges), "\n")

# 3. Create vertex list for these birds
birds <- unique(c(edges$bird1, edges$bird2))
vertices <- data.frame(individual = birds, stringsAsFactors = FALSE)
vertices <- vertices %>%
  left_join(degree_long %>% filter(year == year), by = c("individual")) %>%
  left_join(betweenness_long %>% filter(year == year), by = c("individual", "year"))

cat("Number of vertices:", nrow(vertices), "\n")

# 4. Create igraph object
g <- graph_from_data_frame(edges, directed = FALSE, vertices = vertices)
E(g)$weight <- edges$dyad_strength
layout <- layout_with_fr(g, weights = E(g)$weight)

# 5. Set node colors (if sex is available)
node_colors <- ifelse(is.null(V(g)$sex) | is.na(V(g)$sex), "gray",
                      ifelse(V(g)$sex == "F", "pink", "lightblue"))

# 6. Set node sizes based on betweenness (if available)
if (!is.null(V(g)$mean_betweenness) && any(!is.na(V(g)$mean_betweenness))) {
  node_sizes <- 3 + (V(g)$mean_betweenness / max(V(g)$mean_betweenness, na.rm = TRUE)) * 15
} else {
  node_sizes <- 5
}

# 7. Plot the network
plot(g, 
     layout = layout,
     vertex.color = node_colors,
     vertex.size = node_sizes,
     vertex.label = V(g)$individual,
     vertex.label.cex = 0.6,
     vertex.label.color = "black",
     edge.width = 1,
     edge.curved = 0.2,
     main = "SPRA42 Colony Network 2017",
     sub = "Node size = Betweenness, Edge width = Dyad strength")

print("Plot should be visible now.")



# ============================================================================
# 3. CREATE BETWEENNESS-CENTRIC VISUALIZATION
# ============================================================================

create_betweenness_viz <- function(year, output_file) {
  
  # Filter data for specific year
  year_dyads <- dyad_data %>% filter(year == !!year)
  
  # Create edge list
  edges <- year_dyads %>%
    select(bird1, bird2, dyad_strength, sex_pair) %>%
    filter(!is.na(dyad_strength) & dyad_strength > 0)
  
  # Create vertex list
  vertices <- individual_data %>%
    filter(year == !!year) %>%
    select(individual, mean_degree, sex) %>%
    distinct()
  
  # Add betweenness centrality
  vertices <- vertices %>%
    left_join(
      betweenness_data %>% 
        filter(year == !!year) %>%
        select(individual, mean_betweenness),
      by = "individual"
    )
  
  # Create igraph object
  g <- graph_from_data_frame(edges, directed = FALSE, vertices = vertices)
  
  # Set edge weights
  E(g)$weight <- edges$dyad_strength
  
  # Calculate layout optimized for betweenness visualization
  layout <- layout_with_fr(g, weights = E(g)$weight)
  
  # Create betweenness-based color gradient
  betweenness_values <- V(g)$mean_betweenness
  betweenness_normalized <- (betweenness_values - min(betweenness_values, na.rm = TRUE)) / 
                           (max(betweenness_values, na.rm = TRUE) - min(betweenness_values, na.rm = TRUE))
  
  # Color nodes by betweenness (red = high, blue = low)
  node_colors <- colorRampPalette(c("lightblue", "red"))(100)[round(betweenness_normalized * 99) + 1]
  
  # Set node sizes based on betweenness
  node_sizes <- 5 + betweenness_normalized * 20
  
  # Set edge transparency based on dyad strength
  edge_alpha <- 0.3 + (E(g)$weight / max(E(g)$weight)) * 0.7
  edge_colors <- paste0("gray", round(edge_alpha * 100))
  
  # Create the plot
  pdf(output_file, width = 14, height = 12)
  
  plot(g, 
       layout = layout,
       vertex.color = node_colors,
       vertex.size = node_sizes,
       vertex.label = V(g)$individual,
       vertex.label.cex = 0.7,
       vertex.label.color = "black",
       vertex.frame.color = "white",
       edge.color = edge_colors,
       edge.width = 1,
       edge.curved = 0.1,
       main = paste("Betweenness Centrality in Weaver Bird Network", year, "\nRed = High betweenness, Blue = Low betweenness"),
       sub = "Node size and color intensity proportional to betweenness centrality")
  
  # Add color scale
  color_scale <- colorRampPalette(c("lightblue", "red"))(10)
  legend("bottomright", 
         legend = c("Low Betweenness", "High Betweenness"),
         fill = c("lightblue", "red"),
         cex = 0.8)
  
  dev.off()
  
  return(g)
}

# ============================================================================
# 4. CREATE SEX PAIR ROLE VISUALIZATION
# ============================================================================

create_sex_pair_viz <- function(year, output_file) {
  
  # Filter data for specific year
  year_dyads <- dyad_data %>% filter(year == !!year)
  
  # Create edge list
  edges <- year_dyads %>%
    select(bird1, bird2, dyad_strength, sex_pair) %>%
    filter(!is.na(dyad_strength) & dyad_strength > 0)
  
  # Create vertex list
  vertices <- individual_data %>%
    filter(year == !!year) %>%
    select(individual, mean_degree, sex) %>%
    distinct()
  
  # Add betweenness centrality
  vertices <- vertices %>%
    left_join(
      betweenness_data %>% 
        filter(year == !!year) %>%
        select(individual, mean_betweenness),
      by = "individual"
    )
  
  # Create igraph object
  g <- graph_from_data_frame(edges, directed = FALSE, vertices = vertices)
  
  # Set edge weights
  E(g)$weight <- edges$dyad_strength
  
  # Calculate layout
  layout <- layout_with_fr(g, weights = E(g)$weight)
  
  # Set node colors based on sex
  node_colors <- ifelse(V(g)$sex == "F", "pink", "lightblue")
  
  # Set node sizes based on betweenness
  node_sizes <- 3 + (V(g)$mean_betweenness / max(V(g)$mean_betweenness, na.rm = TRUE)) * 15
  
  # Create edge colors and styles based on sex pair
  edge_colors <- case_when(
    edges$sex_pair == "ff" ~ "red",
    edges$sex_pair == "mm" ~ "blue", 
    edges$sex_pair == "fm" ~ "green",
    edges$sex_pair == "mf" ~ "orange",
    TRUE ~ "gray"
  )
  
  # Set edge widths based on dyad strength
  edge_widths <- 0.5 + (E(g)$weight / max(E(g)$weight)) * 4
  
  # Create the plot
  pdf(output_file, width = 16, height = 12)
  
  # Create multi-panel layout
  par(mfrow = c(2, 2))
  
  # Panel 1: Full network
  plot(g, 
       layout = layout,
       vertex.color = node_colors,
       vertex.size = node_sizes,
       vertex.label = V(g)$individual,
       vertex.label.cex = 0.6,
       vertex.label.color = "black",
       edge.color = edge_colors,
       edge.width = edge_widths,
       edge.curved = 0.2,
       main = paste("Full Network", year))
  
  # Panel 2: FF dyads only
  ff_edges <- edges %>% filter(sex_pair == "ff")
  if(nrow(ff_edges) > 0) {
    g_ff <- graph_from_data_frame(ff_edges, directed = FALSE, vertices = vertices)
    E(g_ff)$weight <- ff_edges$dyad_strength
    layout_ff <- layout_with_fr(g_ff, weights = E(g_ff)$weight)
    
    plot(g_ff, 
         layout = layout_ff,
         vertex.color = node_colors,
         vertex.size = node_sizes,
         vertex.label = V(g_ff)$individual,
         vertex.label.cex = 0.6,
         vertex.label.color = "black",
         edge.color = "red",
         edge.width = 2,
         edge.curved = 0.2,
         main = "FF Dyads (Anchor Nodes)")
  }
  
  # Panel 3: MF/FM dyads only
  mf_fm_edges <- edges %>% filter(sex_pair %in% c("mf", "fm"))
  if(nrow(mf_fm_edges) > 0) {
    g_mf_fm <- graph_from_data_frame(mf_fm_edges, directed = FALSE, vertices = vertices)
    E(g_mf_fm)$weight <- mf_fm_edges$dyad_strength
    layout_mf_fm <- layout_with_fr(g_mf_fm, weights = E(g_mf_fm)$weight)
    
    plot(g_mf_fm, 
         layout = layout_mf_fm,
         vertex.color = node_colors,
         vertex.size = node_sizes,
         vertex.label = V(g_mf_fm)$individual,
         vertex.label.cex = 0.6,
         vertex.label.color = "black",
         edge.color = ifelse(mf_fm_edges$sex_pair == "fm", "green", "orange"),
         edge.width = 2,
         edge.curved = 0.2,
         main = "MF/FM Dyads (Bonded Pairs)")
  }
  
  # Panel 4: MM dyads only
  mm_edges <- edges %>% filter(sex_pair == "mm")
  if(nrow(mm_edges) > 0) {
    g_mm <- graph_from_data_frame(mm_edges, directed = FALSE, vertices = vertices)
    E(g_mm)$weight <- mm_edges$dyad_strength
    layout_mm <- layout_with_fr(g_mm, weights = E(g_mm)$weight)
    
    plot(g_mm, 
         layout = layout_mm,
         vertex.color = node_colors,
         vertex.size = node_sizes,
         vertex.label = V(g_mm)$individual,
         vertex.label.cex = 0.6,
         vertex.label.color = "black",
         edge.color = "blue",
         edge.width = 2,
         edge.curved = 0.2,
         main = "MM Dyads")
  }
  
  dev.off()
  
  return(g)
}

# ============================================================================
# 5. CREATE AGGREGATED NETWORK VISUALIZATION
# ============================================================================

create_aggregated_viz <- function(output_file) {
  
  # Load aggregated network data
  if(file.exists("network_photos/network_aggregated.graphml")) {
    g <- read_graph("network_photos/network_aggregated.graphml", format = "graphml")
  } else {
    cat("Aggregated network file not found. Creating from dyad data...\n")
    
    # Create aggregated network from all years
    all_dyads <- dyad_data %>%
      select(bird1, bird2, dyad_strength, sex_pair) %>%
      filter(!is.na(dyad_strength) & dyad_strength > 0) %>%
      group_by(bird1, bird2, sex_pair) %>%
      summarise(avg_strength = mean(dyad_strength, na.rm = TRUE), .groups = 'drop')
    
    # Create igraph object
    g <- graph_from_data_frame(all_dyads, directed = FALSE)
    E(g)$weight <- all_dyads$avg_strength
    E(g)$sex_pair <- all_dyads$sex_pair
  }
  
  # Calculate betweenness centrality
  betweenness_values <- betweenness(g, weights = E(g)$weight)
  
  # Set node sizes based on betweenness
  node_sizes <- 3 + (betweenness_values / max(betweenness_values)) * 20
  
  # Set node colors based on betweenness (gradient)
  betweenness_normalized <- betweenness_values / max(betweenness_values)
  node_colors <- colorRampPalette(c("lightblue", "red"))(100)[round(betweenness_normalized * 99) + 1]
  
  # Set edge colors based on sex pair
  edge_colors <- case_when(
    E(g)$sex_pair == "ff" ~ "red",
    E(g)$sex_pair == "mm" ~ "blue", 
    E(g)$sex_pair == "fm" ~ "green",
    E(g)$sex_pair == "mf" ~ "orange",
    TRUE ~ "gray"
  )
  
  # Set edge widths based on strength
  edge_widths <- 0.5 + (E(g)$weight / max(E(g)$weight)) * 3
  
  # Calculate layout
  layout <- layout_with_fr(g, weights = E(g)$weight)
  
  # Create the plot
  pdf(output_file, width = 16, height = 14)
  
  plot(g, 
       layout = layout,
       vertex.color = node_colors,
       vertex.size = node_sizes,
       vertex.label = V(g)$name,
       vertex.label.cex = 0.5,
       vertex.label.color = "black",
       vertex.frame.color = "white",
       edge.color = edge_colors,
       edge.width = edge_widths,
       edge.curved = 0.2,
       main = "Aggregated Weaver Bird Social Network\nNode color/size = Betweenness centrality, Edge color = Sex pair type",
       sub = "Red edges = FF (anchor nodes), Green/Orange = FM/MF (bonded pairs), Blue = MM")
  
  # Add comprehensive legend
  legend("bottomright", 
         legend = c("High Betweenness", "Low Betweenness", "FF dyad", "MM dyad", "FM dyad", "MF dyad"),
         col = c("red", "lightblue", "red", "blue", "green", "orange"),
         pch = c(19, 19, 15, 15, 15, 15),
         pt.cex = c(2, 2, 1, 1, 1, 1),
         cex = 0.8)
  
  dev.off()
  
  return(g)
}

# ============================================================================
# 6. EXECUTE VISUALIZATIONS
# ============================================================================

cat("Creating network visualizations...\n")

# Create directory for visualizations
dir.create("network_visualizations", showWarnings = FALSE)

# Create visualizations for each year
years <- unique(dyad_data$year)
for(year in years) {
  cat("Creating visualizations for year", year, "...\n")
  
  # Full network
  create_network_viz(year)
  
  # Betweenness-focused
  create_betweenness_viz(year, paste0("network_visualizations/betweenness_", year, ".pdf"))
  
  # Sex pair roles
  create_sex_pair_viz(year, paste0("network_visualizations/sex_pairs_", year, ".pdf"))
}

# Create aggregated network
cat("Creating aggregated network visualization...\n")
create_aggregated_viz("network_visualizations/aggregated_network.pdf")

cat("Network visualizations complete! Check the network_visualizations/ folder.\n") 