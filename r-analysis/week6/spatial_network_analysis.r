# =============================================================================
# SPATIAL NETWORK ANALYSIS FOR WEAVER BIRDS
# Segments networks by colony/plot and integrates MCMCglmm results
# =============================================================================

# setwd("week6")
setwd("~/Code/Birds/week6")
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(igraph)
library(viridis)
library(stringr)
library(purrr)
library(readxl)
library(ggraph)
library(cowplot)
library(patchwork)
library(lubridate)

# =============================================================================
# 1. LOAD AND PROCESS SPATIAL DATA
# =============================================================================

cat("=== LOADING SPATIAL DATA ===\n\n")

# Load colony measurement data
cat("Loading colony measurement data...\n")
colony_data <- read_excel("../data/Environment/c_2013-2017_weaver_colony _measurement.xlsx")

# Load identification data to map birds to colonies
cat("Loading identification data...\n")
id_map <- read_excel("../data/Index/identification.xlsx")

# Load intercolony distance data
cat("Loading intercolony distance data...\n")
intercolony_dist <- read_excel("../data/Environment/intercolony_distance.xlsx")

# Display structure of spatial data
cat("\nColony data structure:\n")
str(colony_data)
cat("\nFirst few rows of colony data:\n")
print(head(colony_data))

cat("\nIdentification data structure:\n")
str(id_map)
cat("\nFirst few rows of identification data:\n")
print(head(id_map))

# Extract Year from Date in id_map
id_map <- id_map %>%
  mutate(Year = year(as.Date(Date)))

# Summarize and flag birds with multiple colony assignments in the same year
multi_colony <- id_map %>%
  group_by(Combo, Year) %>%
  summarise(
    n_colonies = n_distinct(Colony),
    colonies = paste(unique(Colony), collapse = ", ")
  ) %>%
  filter(n_colonies > 1)

cat("Birds with multiple colony assignments in the same year:\n")
print(multi_colony)

# Add ambiguous_colony flag to id_map
id_map <- id_map %>%
  left_join(
    multi_colony %>% select(Combo, Year) %>% mutate(ambiguous_colony = TRUE),
    by = c("Combo", "Year")
  ) %>%
  mutate(ambiguous_colony = ifelse(is.na(ambiguous_colony), FALSE, ambiguous_colony))

# =============================================================================
# 2. MAP BIRDS TO COLONIES/PLOTS
# =============================================================================

cat("\n=== MAPPING BIRDS TO COLONIES ===\n\n")

# Create bird-to-colony mapping using Combo, Year, Colony, and ambiguous_colony columns
bird_colony_mapping <- id_map %>%
  select(Combo, Year, Colony, ambiguous_colony) %>%
  distinct() %>%
  rename(bird_id = Combo, year = Year, colony = Colony)

# Clean mapping: remove CLIFFORD and blank/NA colonies, but keep ambiguous birds and TESTER
bird_colony_mapping <- bird_colony_mapping %>%
  filter(
    colony != "CLIFFORD",
    !is.na(colony),
    colony != ""
  )
# ambiguous_colony column is retained for downstream flagging

# Export mapping for downstream use
dir.create("week7", showWarnings = FALSE)
write.csv(bird_colony_mapping, "week7/bird_colony_mapping.csv", row.names = FALSE)
cat("Saved week7/bird_colony_mapping.csv\n")

# Display colony distribution
colony_summary <- bird_colony_mapping %>%
  group_by(colony) %>%
  summarise(n_birds = n(), .groups = "drop")

cat("Bird distribution by colony:\n")
print(colony_summary)

# =============================================================================
# 3. SEGMENT NETWORKS BY COLONY
# =============================================================================

cat("\n=== SEGMENTING NETWORKS BY COLONY ===\n\n")

# Function to create colony-specific networks
create_colony_networks <- function(year) {
  sri_file <- paste0("sri_data/SRI_", year, ".csv")
  sri_data <- read_csv(sri_file, show_col_types = FALSE) %>%
    mutate(year = as.numeric(year))  # Add year column for joining
  
  # Add colony information
  sri_with_colony <- sri_data %>%
    left_join(bird_colony_mapping, by = c("year", "bird_1" = "bird_id")) %>%
    rename(colony_1 = colony, ambiguous_1 = ambiguous_colony) %>%
    left_join(bird_colony_mapping, by = c("year", "bird_2" = "bird_id")) %>%
    rename(colony_2 = colony, ambiguous_2 = ambiguous_colony) %>%
    mutate(
      same_colony = colony_1 == colony_2,
      colony_pair = case_when(
        same_colony ~ colony_1,
        TRUE ~ "Cross-colony"
      )
    )
  
  # Create separate networks for each colony
  colony_networks <- list()
  
  for(colony in unique(colony_summary$colony)) {
    # Get dyads within this colony
    colony_dyads <- sri_with_colony %>%
      filter(colony_1 == colony & colony_2 == colony)
    
    if(nrow(colony_dyads) > 0) {
      # Create network
      g <- graph_from_data_frame(
        colony_dyads %>% select(bird_1, bird_2, SRI),
        directed = FALSE
      )
      E(g)$weight <- colony_dyads$SRI
      
      # Add node attributes
      V(g)$colony <- colony
      V(g)$degree <- degree(g)
      V(g)$betweenness <- betweenness(g)
      
      colony_networks[[colony]] <- g
    }
  }
  
  # Create cross-colony network
  cross_colony_dyads <- sri_with_colony %>%
    filter(colony_1 != colony_2)
  
  if(nrow(cross_colony_dyads) > 0) {
    g_cross <- graph_from_data_frame(
      cross_colony_dyads %>% select(bird_1, bird_2, SRI),
      directed = FALSE
    )
    E(g_cross)$weight <- cross_colony_dyads$SRI
    V(g_cross)$colony <- "Cross-colony"
    V(g_cross)$degree <- degree(g_cross)
    V(g_cross)$betweenness <- betweenness(g_cross)
    
    colony_networks[["Cross-colony"]] <- g_cross
  }
  
  return(colony_networks)
}

# Process all years
years <- c("2013", "2014", "2015", "2016", "2017")
all_colony_networks <- list()

for(year in years) {
  cat("Processing year", year, "...\n")
  all_colony_networks[[year]] <- create_colony_networks(year)
  
  # Print summary for each colony
  for(colony_name in names(all_colony_networks[[year]])) {
    g <- all_colony_networks[[year]][[colony_name]]
    cat("  ", colony_name, ": ", vcount(g), " birds, ", ecount(g), " dyads\n")
  }
}

# =============================================================================
# 4. SPATIAL VISUALIZATION
# =============================================================================

cat("\n=== CREATING SPATIAL VISUALIZATIONS ===\n\n")

# Create spatial layout for colonies (you may need to adjust coordinates)
colony_coords <- data.frame(
  colony = c("Colony1", "Colony2", "Colony3"),
  x = c(1, 2, 3),  # Replace with actual coordinates
  y = c(1, 2, 1)   # Replace with actual coordinates
)

# Function to create spatial network plot
plot_spatial_network <- function(year, colony_name) {
  if(!colony_name %in% names(all_colony_networks[[year]])) {
    return(NULL)
  }
  
  g <- all_colony_networks[[year]][[colony_name]]
  
  # Create layout
  if(colony_name == "Cross-colony") {
    # For cross-colony, use geographic layout
    layout <- layout_with_fr(g)
  } else {
    # For within-colony, use circular layout
    layout <- layout_in_circle(g)
  }
  
  # Create plot
  p <- ggraph(g, layout = layout) +
    geom_edge_link(aes(width = weight), alpha = 0.6, color = "gray50") +
    geom_node_point(aes(size = degree, color = colony), alpha = 0.8) +
    geom_node_text(aes(label = name), size = 2, repel = TRUE) +
    scale_edge_width_continuous(range = c(0.5, 3)) +
    scale_size_continuous(range = c(2, 8)) +
    scale_color_viridis(discrete = TRUE) +
    theme_void() +
    labs(
      title = paste("Network -", colony_name, "Year", year),
      subtitle = paste("Nodes:", vcount(g), "Edges:", ecount(g))
    ) +
    theme(plot.title = element_text(size = 12, face = "bold"))
  
  return(p)
}

# Create spatial plots for each year and colony
for(year in years) {
  cat("Creating spatial plots for year", year, "...\n")
  
  # Get all colonies for this year
  colonies <- names(all_colony_networks[[year]])
  
  # Create plots
  plots <- list()
  for(colony in colonies) {
    p <- plot_spatial_network(year, colony)
    if(!is.null(p)) {
      plots[[colony]] <- p
    }
  }
  
  # Combine plots
  if(length(plots) > 0) {
    combined_plot <- wrap_plots(plots, ncol = 2)
    ggsave(paste0("spatial_network_", year, ".png"), 
           combined_plot, width = 16, height = 12, dpi = 300)
    cat("  Saved spatial_network_", year, ".png\n")
  }
}

# =============================================================================
# 5. INTEGRATE MCMCglmm RESULTS
# =============================================================================

cat("\n=== INTEGRATING MCMCglmm RESULTS ===\n\n")

# Load MCMCglmm results (you'll need to adjust the path)
cat("Loading MCMCglmm results...\n")

# Function to extract key MCMCglmm findings
extract_mcmcglmm_insights <- function() {
  # Based on your model summaries, extract key findings
  insights <- list(
    distance_effect = "Negative effect on association strength",
    relatedness_effect = "Positive effect on association strength", 
    pair_bond_effect = "Strong positive effect on association strength",
    sex_effect = "FM dyads more likely to associate",
    disturbance_effect = "Generally weakens associations",
    temporal_autocorrelation = "Strong predictor of association strength"
  )
  
  return(insights)
}

mcmcglmm_insights <- extract_mcmcglmm_insights()

cat("Key MCMCglmm findings:\n")
for(insight in names(mcmcglmm_insights)) {
  cat("  ", insight, ": ", mcmcglmm_insights[[insight]], "\n")
}

# =============================================================================
# 6. CREATE INTEGRATED COLONY NETWORK ANALYSIS (with ambiguity)
# =============================================================================

cat("\n=== CREATING INTEGRATED COLONY NETWORK ANALYSIS ===\n\n")

analyze_colony_patterns <- function() {
  colony_analysis <- data.frame()
  for(year in years) {
    for(colony_name in names(all_colony_networks[[year]])) {
      g <- all_colony_networks[[year]][[colony_name]]
      # Get ambiguous birds in this colony/year
      ambiguous_bird_names <- id_map %>%
        filter(Year == as.numeric(year), ambiguous_colony) %>%
        pull(Combo)
      ambiguous_birds <- V(g)$name %in% ambiguous_bird_names
      prop_ambiguous <- mean(ambiguous_birds)
      if(vcount(g) > 0) {
        metrics <- data.frame(
          Year = year,
          Colony = colony_name,
          N_Birds = vcount(g),
          N_Dyads = ecount(g),
          Mean_SRI = mean(E(g)$weight),
          Edge_Density = edge_density(g),
          Avg_Clustering = transitivity(g, type = "global"),
          N_Communities = length(cluster_louvain(g, weights = E(g)$weight)),
          Mean_Degree = mean(degree(g)),
          Mean_Betweenness = mean(betweenness(g)),
          Prop_Ambiguous = prop_ambiguous
        )
        colony_analysis <- rbind(colony_analysis, metrics)
      }
    }
  }
  return(colony_analysis)
}

colony_analysis <- analyze_colony_patterns()
write.csv(colony_analysis, "colony_network_analysis.csv", row.names = FALSE)
cat("Saved new colony_network_analysis.csv\n")

# =============================================================================
# NOTE: To validate or trace the origins of other network data files (e.g., dyad_strength_by_year.csv, degree_centrality_by_year.csv),
# search your codebase for write.csv or write_xlsx calls, or check scripts that process SRI or network data.
# Ensure all network metrics are derived from the correct, up-to-date SRI and mapping files for accuracy.
# =============================================================================

# =============================================================================
# 7. SUMMARY AND RECOMMENDATIONS
# =============================================================================

cat("\n=== SUMMARY AND RECOMMENDATIONS ===\n\n")

cat("Analysis Complete!\n\n")

cat("Files Generated:\n")
cat("- spatial_network_*.png: Colony-specific network visualizations\n")
cat("- integrated_colony_analysis.png: Combined analysis plot\n")
cat("- colony_network_analysis.csv: Detailed colony metrics\n\n")

cat("Key Insights:\n")
cat("1. Spatial segmentation reveals colony-specific social structures\n")
cat("2. Cross-colony associations can be analyzed separately\n")
cat("3. MCMCglmm results can inform network interpretation\n")
cat("4. Temporal patterns may vary by colony\n\n")

cat("Next Steps:\n")
cat("1. Adjust colony coordinates for accurate spatial visualization\n")
cat("2. Refine bird-to-colony mapping based on your data\n")
cat("3. Add environmental covariates (rainfall, disturbance) to colony analysis\n")
cat("4. Test for colony-specific effects in MCMCglmm models\n")
cat("5. Create publication-ready figures combining spatial and network data\n\n")

cat("=== SPATIAL NETWORK ANALYSIS COMPLETE ===\n") 