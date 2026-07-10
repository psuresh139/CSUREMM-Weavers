# =============================================================================
# NETWORK ANALYSIS AND GRAPHING FOR WEAVER BIRD SOCIAL NETWORKS
# R Replica of math_modeling.py functionality
# =============================================================================

# Load required packages

library(igraph)
library(readr)
library(ggplot2)
library(ggraph)
library(dplyr)

library(tidyr)
library(stringr)
library(purrr)
library(viridis)
library(cowplot)
library(patchwork)
setwd("~/Code/Birds")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

#' Build igraph object from SRI CSV file
#' @param csv_path Path to CSV file with columns: bird_1, bird_2, SRI
#' @return igraph object with nodes and weighted edges
build_igraph_from_sri <- function(csv_path) {
  # Read CSV
  df <- read_csv(csv_path, show_col_types = FALSE)
  
  # Get unique nodes
  nodes <- unique(c(df$bird_1, df$bird_2))
  
  # Create edge list with weights
  edges <- df %>%
    select(bird_1, bird_2, SRI) %>%
    mutate(
      from = bird_1,
      to = bird_2
    ) %>%
    select(from, to, SRI)
  
  # Create igraph object
  g <- graph_from_data_frame(edges, directed = FALSE, vertices = data.frame(name = nodes))
  
  # Set edge weights
  E(g)$weight <- edges$SRI
  
  return(g)
}

#' Compute network metrics for a given graph
#' @param g igraph object
#' @return List of computed metrics
compute_metrics <- function(g) {
  # Basic metrics
  degree_centrality <- degree(g)
  betweenness <- betweenness(g)
  clustering <- transitivity(g, type = "local", isolates = "zero")
  edge_density <- edge_density(g)
  
  # Community detection using Louvain method
  communities <- cluster_louvain(g, weights = E(g)$weight)
  
  return(list(
    degree_centrality = degree_centrality,
    betweenness = betweenness,
    clustering = clustering,
    edge_density = edge_density,
    n_communities = length(communities),
    communities = communities,
    avg_clustering = mean(clustering, na.rm = TRUE)
  ))
}

#' Plot network and save to file
#' @param g igraph object
#' @param year Year identifier for filename
#' @param save_plot Whether to save plot (default: TRUE)
plot_network <- function(g, year, save_plot = TRUE) {
  # Create layout
  layout <- layout_with_fr(g)
  
  # Create plot using ggraph
  p <- ggraph(g, layout = layout) +
    geom_edge_link(aes(width = weight), alpha = 0.6, color = "gray50") +
    geom_node_point(size = 3, color = "steelblue") +
    geom_node_text(aes(label = name), size = 2, repel = TRUE) +
    scale_edge_width_continuous(range = c(0.5, 3)) +
    theme_void() +
    labs(title = paste("Social Network - Year", year),
         subtitle = paste("Nodes:", vcount(g), "Edges:", ecount(g))) +
    theme(plot.title = element_text(size = 14, face = "bold"))
  
  if(save_plot) {
    ggsave(paste0("week7/network_", year, ".png"), p, width = 8, height = 6, dpi = 300)
    cat("Saved week7/network_", year, ".png\n", sep = "")
  }
  
  return(p)
}

#' Plot evolution of network metrics over time
#' @param years Vector of years
#' @param all_metrics List of metrics for each year
#' @param all_graphs List of graphs for each year
plot_metrics_evolution <- function(years, all_metrics, all_graphs) {
  # Prepare data
  years_sorted <- sort(years)
  edge_density <- sapply(years_sorted, function(y) all_metrics[[y]]$edge_density)
  avg_clustering <- sapply(years_sorted, function(y) all_metrics[[y]]$avg_clustering)
  n_communities <- sapply(years_sorted, function(y) all_metrics[[y]]$n_communities)
  
  # Create data frame for plotting
  plot_data <- data.frame(
    Year = years_sorted,
    Edge_Density = edge_density,
    Avg_Clustering = avg_clustering,
    N_Communities = n_communities
  ) %>%
    pivot_longer(-Year, names_to = "Metric", values_to = "Value")
  
  # Create plot
  p <- ggplot(plot_data, aes(x = Year, y = Value, color = Metric, group = Metric)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    scale_color_viridis(discrete = TRUE, option = "D") +
    labs(title = "Evolution of Network Structure Over Time",
         x = "Year",
         y = "Metric Value",
         color = "Metric") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  
  # Save plot
  ggsave("week7/network_metrics_evolution.png", p, width = 10, height = 6, dpi = 300)
  cat("Saved week7/network_metrics_evolution.png\n")
  
  return(p)
}

#' Track degree centrality for individuals across years
#' @param years Vector of years
#' @param all_graphs List of graphs for each year
track_degree_centrality <- function(years, all_graphs) {
  # Collect all unique node names
  all_names <- unique(unlist(lapply(all_graphs, function(g) V(g)$name)))
  
  # Build data frame
  degree_df <- matrix(0, nrow = length(all_names), ncol = length(years))
  rownames(degree_df) <- all_names
  colnames(degree_df) <- years
  
  for(year in years) {
    g <- all_graphs[[year]]
    degrees <- degree(g)
    degree_df[V(g)$name, year] <- degrees
  }
  
  # Convert to data frame and save
  degree_df <- as.data.frame(degree_df)
  degree_df$individual <- rownames(degree_df)
  write_csv(degree_df, "week7/degree_centrality_by_year.csv")
  cat("Saved week7/degree_centrality_by_year.csv\n")
  
  # Identify top changers
  degree_change <- apply(degree_df[, years], 1, function(x) max(x) - min(x))
  top_changers <- sort(degree_change, decreasing = TRUE)[1:10]
  
  cat("Top 10 individuals with largest change in degree centrality:\n")
  print(top_changers)
  
  return(degree_df)
}

#' Track betweenness centrality for individuals across years
#' @param years Vector of years
#' @param all_graphs List of graphs for each year
track_betweenness_centrality <- function(years, all_graphs) {
  # Collect all unique node names
  all_names <- unique(unlist(lapply(all_graphs, function(g) V(g)$name)))
  
  # Build data frame
  betweenness_df <- matrix(0, nrow = length(all_names), ncol = length(years))
  rownames(betweenness_df) <- all_names
  colnames(betweenness_df) <- years
  
  for(year in years) {
    g <- all_graphs[[year]]
    betweenness_vals <- betweenness(g)
    betweenness_df[V(g)$name, year] <- betweenness_vals
  }
  
  # Convert to data frame and save
  betweenness_df <- as.data.frame(betweenness_df)
  betweenness_df$individual <- rownames(betweenness_df)
  write_csv(betweenness_df, "week7/betweenness_centrality_by_year.csv")
  cat("Saved week7/betweenness_centrality_by_year.csv\n")
  
  # Identify top changers
  betweenness_change <- apply(betweenness_df[, years], 1, function(x) max(x) - min(x))
  top_changers <- sort(betweenness_change, decreasing = TRUE)[1:10]
  
  cat("Top 10 individuals with largest change in betweenness centrality:\n")
  print(top_changers)
  
  return(betweenness_df)
}

#' Track dyad strength (SRI) across years
#' @param years Vector of years
#' @param all_graphs List of graphs for each year
track_dyad_strength <- function(years, all_graphs) {
  # Collect all unique dyads
  all_dyads <- list()
  for(g in all_graphs) {
    edges <- as_edgelist(g)
    for(i in 1:nrow(edges)) {
      dyad <- sort(c(edges[i, 1], edges[i, 2]))
      all_dyads[[paste(dyad[1], dyad[2], sep = "_")]] <- dyad
    }
  }
  
  # Create unique dyad list
  unique_dyads <- unique(do.call(rbind, all_dyads))
  
  # Build data frame
  dyad_df <- matrix(0, nrow = nrow(unique_dyads), ncol = length(years))
  rownames(dyad_df) <- apply(unique_dyads, 1, function(x) paste(x[1], x[2], sep = "_"))
  colnames(dyad_df) <- years
  
  for(year in years) {
    g <- all_graphs[[year]]
    edges <- as_edgelist(g)
    weights <- E(g)$weight
    
    for(i in 1:nrow(edges)) {
      dyad <- sort(c(edges[i, 1], edges[i, 2]))
      dyad_key <- paste(dyad[1], dyad[2], sep = "_")
      if(dyad_key %in% rownames(dyad_df)) {
        dyad_df[dyad_key, year] <- weights[i]
      }
    }
  }
  
  # Convert to data frame and save
  dyad_df <- as.data.frame(dyad_df)
  dyad_df$dyad <- rownames(dyad_df)
  write_csv(dyad_df, "week7/dyad_strength_by_year.csv")
  cat("Saved week7/dyad_strength_by_year.csv\n")
  
  # Identify top changers
  dyad_change <- apply(dyad_df[, years], 1, function(x) max(x) - min(x))
  top_changers <- sort(dyad_change, decreasing = TRUE)[1:10]
  
  cat("Top 10 dyads with largest change in SRI:\n")
  print(top_changers)
  
  return(dyad_df)
}

#' Export networks to GraphML format
#' @param years Vector of years
#' @param all_graphs List of graphs for each year
export_networks_to_graphml <- function(years, all_graphs) {
  for(year in years) {
    g <- all_graphs[[year]]
    
    # Add node attributes
    V(g)$degree <- degree(g)
    V(g)$betweenness <- betweenness(g)
    
    # Write GraphML file
    out_path <- paste0("week7/network_", year, ".graphml")
    write_graph(g, out_path, format = "graphml")
    cat("Saved", out_path, "\n")
  }
}

#' Build aggregated network across all years
#' @param years Vector of years
#' @param all_graphs List of graphs for each year
#' @return Aggregated igraph object
build_aggregated_network <- function(years, all_graphs) {
  # Collect all unique dyads and their weights
  all_edges <- list()
  
  for(year in years) {
    g <- all_graphs[[year]]
    edges <- as_edgelist(g)
    weights <- E(g)$weight
    
    for(i in 1:nrow(edges)) {
      dyad <- sort(c(edges[i, 1], edges[i, 2]))
      dyad_key <- paste(dyad[1], dyad[2], sep = "_")
      
      if(is.null(all_edges[[dyad_key]])) {
        all_edges[[dyad_key]] <- list(dyad = dyad, weights = numeric(length(years)))
      }
      year_idx <- which(years == year)
      all_edges[[dyad_key]]$weights[year_idx] <- weights[i]
    }
  }
  
  # Create aggregated network
  unique_nodes <- unique(unlist(lapply(all_graphs, function(g) V(g)$name)))
  
  # Calculate mean weights for each dyad
  edge_list <- do.call(rbind, lapply(all_edges, function(x) {
    mean_weight <- mean(x$weights[x$weights > 0], na.rm = TRUE)
    if(is.na(mean_weight)) mean_weight <- 0
    c(x$dyad, mean_weight)
  }))
  
  # Create igraph object
  g_agg <- graph_from_data_frame(edge_list, directed = FALSE, 
                                vertices = data.frame(name = unique_nodes))
  E(g_agg)$weight <- as.numeric(edge_list[, 3])
  
  return(g_agg)
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Find all SRI CSV files
sri_files <- list.files(path = "week7/", pattern = "SRI_.*\\.csv", full.names = TRUE)
if(length(sri_files) == 0) {
  cat("No SRI_*.csv files found in week7/.\n")
  cat("Please ensure SRI files are in the week7/ directory.\n")
  stop("No data files found")
}

# Extract years from filenames
years <- str_extract(basename(sri_files), "(?<=SRI_)[0-9]+(?=\\.csv)")
years <- sort(years)

cat("Found", length(years), "years of data:", paste(years, collapse = ", "), "\n")

# Process all files
all_metrics <- list()
all_graphs <- list()

for(i in seq_along(sri_files)) {
  year <- years[i]
  file_path <- sri_files[i]
  
  cat("Processing year", year, "from", basename(file_path), "\n")
  
  # Build graph
  g <- build_igraph_from_sri(file_path)
  all_graphs[[year]] <- g
  
  # Compute metrics
  metrics <- compute_metrics(g)
  all_metrics[[year]] <- metrics
  
  # Print summary
  cat("  Nodes:", vcount(g), "Edges:", ecount(g), "\n")
  cat("  Edge density:", round(metrics$edge_density, 3), "\n")
  cat("  Avg clustering:", round(metrics$avg_clustering, 3), "\n")
  cat("  Communities:", metrics$n_communities, "\n\n")
}

# Generate visualizations
cat("Generating network visualizations...\n")
for(year in years) {
  plot_network(all_graphs[[year]], year)
}

# Plot metrics evolution
cat("Plotting metrics evolution...\n")
plot_metrics_evolution(years, all_metrics, all_graphs)

# Track individual metrics
cat("Tracking individual metrics...\n")
degree_df <- track_degree_centrality(years, all_graphs)
betweenness_df <- track_betweenness_centrality(years, all_graphs)

# Track dyad metrics
cat("Tracking dyad metrics...\n")
dyad_df <- track_dyad_strength(years, all_graphs)

# Export networks
cat("Exporting networks to GraphML...\n")
export_networks_to_graphml(years, all_graphs)

# Build aggregated network
cat("Building aggregated network...\n")
g_agg <- build_aggregated_network(years, all_graphs)
metrics_agg <- compute_metrics(g_agg)

cat("Aggregated network summary:\n")
cat("  Nodes:", vcount(g_agg), "Edges:", ecount(g_agg), "\n")
cat("  Edge density:", round(metrics_agg$edge_density, 3), "\n")
cat("  Avg clustering:", round(metrics_agg$avg_clustering, 3), "\n")
cat("  Communities:", metrics_agg$n_communities, "\n")

# Save aggregated network
write_graph(g_agg, "week7/network_aggregated.graphml", format = "graphml")
cat("Saved week7/network_aggregated.graphml\n")

# Create summary report
cat("\n=== NETWORK ANALYSIS COMPLETE ===\n")
cat("Files generated:\n")
cat("- network_*.png: Individual year network plots\n")
cat("- network_metrics_evolution.png: Temporal trends\n")
cat("- degree_centrality_by_year.csv: Individual degree tracking\n")
cat("- betweenness_centrality_by_year.csv: Individual betweenness tracking\n")
cat("- dyad_strength_by_year.csv: Dyad SRI tracking\n")
cat("- network_*.graphml: Network files for external analysis\n")
cat("- network_aggregated.graphml: Multi-year aggregated network\n")

cat("\nAnalysis complete!\n")

# After all network metrics and community detection are computed, output a new colony_network_analysis.csv
# This should include: year, id (bird), colony (standardized), community

# Example: assuming you have a list all_graphs (one per year), and a data frame or list mapping birds to colonies per year
# You may need to adapt this to your actual variable names and structures

# Read the bird-to-colony mapping
bird_colony_mapping <- read.csv("week7/bird_colony_mapping.csv", stringsAsFactors = FALSE)

colony_network_analysis <- data.frame()
for (year in years) {
  g <- all_graphs[[year]]
  comm <- igraph::vertex_attr(g, 'community')
  ids <- igraph::vertex_attr(g, 'name')
  # Assign colony for each bird in this year
  year_colony <- sapply(ids, function(b) {
    match_row <- which(bird_colony_mapping$bird_id == b & bird_colony_mapping$year == as.numeric(year))
    if (length(match_row) > 0) bird_colony_mapping$colony[match_row[1]] else NA
  })
  year_vec <- rep(year, length(ids))
  df <- data.frame(year = year_vec, id = ids, colony = year_colony, community = comm)
  colony_network_analysis <- rbind(colony_network_analysis, df)
}
# Standardize colony names (optional)
colony_network_analysis$colony <- as.character(colony_network_analysis$colony)
colony_network_analysis$colony[colony_network_analysis$colony == '' | is.na(colony_network_analysis$colony)] <- 'unknown'
# Write to CSV
write.csv(colony_network_analysis, 'week7/colony_network_analysis.csv', row.names = FALSE)
cat('Saved week7/colony_network_analysis.csv\n') 