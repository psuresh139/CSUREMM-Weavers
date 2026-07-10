# Connector Bird and Supercolony Analysis
# Focused analysis of connector birds (4-letter IDs) and their role in supercolonies
# Date: 2024

# Load required libraries
library(dplyr)
library(ggplot2)
library(readxl)

# Set working directory
setwd("/Users/pranavsuresh/Code/Birds/week7")

cat("=== CONNECTOR BIRD AND SUPERCOLONY ANALYSIS ===\n\n")

# ============================================================================
# PART 1: LOAD AND PREPARE DATA
# ============================================================================

cat("Loading data...\n")

# Load the main dyad data (this worked for FF analysis)
dyad_data <- read.csv("dyad_full_integrated.csv", stringsAsFactors = FALSE)
cat("Loaded dyad data with", nrow(dyad_data), "rows\n")

# ============================================================================
# PART 2: IDENTIFY CONNECTOR BIRDS FROM DYAD DATA
# ============================================================================

cat("\n=== IDENTIFYING CONNECTOR BIRDS ===\n")

# Extract unique birds from dyad data
all_birds <- unique(c(dyad_data$bird1, dyad_data$bird2))

# Find connector birds (4-letter IDs)
connector_bird_ids <- all_birds[grepl("^[A-Z]{4}$", all_birds)]
cat("Found", length(connector_bird_ids), "connector bird IDs\n")

# Create connector bird summary from dyad data
connector_summary <- data.frame()

for(bird_id in connector_bird_ids) {
  # Get all dyads involving this bird
  bird_dyads <- dyad_data %>%
    filter(bird1 == bird_id | bird2 == bird_id)
  
  if(nrow(bird_dyads) > 0) {
    bird_summary <- bird_dyads %>%
      summarise(
        individual_id = bird_id,
        n_years = n_distinct(year),
        n_colonies = n_distinct(colony),
        n_dyads = n(),
        avg_dyad_strength = mean(dyad_strength, na.rm = TRUE),
        avg_degree = mean((degree_bird1 + degree_bird2) / 2, na.rm = TRUE),
        avg_betweenness = mean((betweenness_bird1 + betweenness_bird2) / 2, na.rm = TRUE)
      )
    
    connector_summary <- rbind(connector_summary, bird_summary)
  }
}

# Filter for true connectors (in multiple colonies)
connector_summary <- connector_summary %>%
  filter(n_colonies > 1) %>%
  mutate(total_connectivity = n_years * n_colonies) %>%
  arrange(desc(total_connectivity))

cat("Found", nrow(connector_summary), "connector birds (in multiple colonies)\n")

# Print top connectors
cat("\nTop 10 connector birds:\n")
print(head(connector_summary, 10))

# ============================================================================
# PART 3: CONNECTOR BIRDS IN SUPERCOLONIES
# ============================================================================

cat("\n=== CONNECTOR BIRDS IN SUPERCOLONIES ===\n")

# Use 'supercolony_id' instead of 'supercolony'
if("supercolony_id" %in% names(dyad_data)) {
  cat("Supercolony data found in dyad data (column: supercolony_id)\n")
  
  # Get supercolony information
  supercolony_info <- dyad_data %>%
    dplyr::select(year, colony, supercolony_id) %>%
    distinct() %>%
    filter(!is.na(supercolony_id) & supercolony_id != "")
  
  cat("Found", n_distinct(supercolony_info$supercolony_id), "unique supercolonies\n")
  
  # Analyze connector birds in supercolonies
  supercolony_connectors <- data.frame()
  
  for(bird_id in connector_bird_ids) {
    # Get dyads for this bird in supercolonies
    bird_dyads <- dyad_data %>%
      filter((bird1 == bird_id | bird2 == bird_id) & !is.na(supercolony_id) & supercolony_id != "")
    
    if(nrow(bird_dyads) > 0) {
      bird_supercolony_summary <- bird_dyads %>%
        group_by(supercolony_id) %>%
        summarise(
          individual_id = bird_id,
          n_years_in_supercolony = n_distinct(year),
          n_dyads_in_supercolony = n(),
          avg_dyad_strength = mean(dyad_strength, na.rm = TRUE)
        ) %>%
        group_by(individual_id) %>%
        summarise(
          n_supercolonies = n(),
          total_years_in_supercolonies = sum(n_years_in_supercolony),
          total_dyads_in_supercolonies = sum(n_dyads_in_supercolony),
          avg_dyad_strength = mean(avg_dyad_strength, na.rm = TRUE)
        )
      
      supercolony_connectors <- rbind(supercolony_connectors, bird_supercolony_summary)
    }
  }
  
  supercolony_connectors <- supercolony_connectors %>%
    arrange(desc(n_supercolonies))
  
  cat("\nConnector birds in supercolonies:\n")
  print(supercolony_connectors)
  
  # If birds_list column exists, cross-reference connector birds using kring mapping
  if("birds_list" %in% names(dyad_data)) {
    cat("\nCross-referencing connector birds with birds_list column using kring mapping...\n")
    # Load mapping file
    kring_map <- read_excel("../data/index/identification.xlsx")
    # Ensure columns are named correctly
    kring_map <- kring_map %>% dplyr::select(Metal, Combo)
    # For each connector bird, get its kring code
    connector_kring_map <- kring_map %>% filter(Combo %in% connector_bird_ids)
    # Get unique supercolony_id and birds_list pairs
    birds_list_info <- dyad_data %>%
      dplyr::select(supercolony_id, birds_list) %>%
      filter(!is.na(supercolony_id) & supercolony_id != "" & !is.na(birds_list) & birds_list != "") %>%
      distinct()
    
    connector_bird_in_supercolony <- data.frame()
    normalize_kring <- function(x) {
      toupper(trimws(x))
    }
    for(i in 1:nrow(connector_kring_map)) {
      kring_id <- normalize_kring(connector_kring_map$Metal[i])
      combo_id <- connector_kring_map$Combo[i]
      in_supercolony <- any(sapply(
        birds_list_info$birds_list,
        function(bl) {
          # Remove brackets and single quotes, then split
          bl_clean <- gsub("\\[|\\]|'", "", bl)
          bl_split <- unlist(strsplit(bl_clean, ","))
          bl_split <- normalize_kring(bl_split)
          kring_id %in% bl_split
        }
      ))
      connector_bird_in_supercolony <- rbind(connector_bird_in_supercolony, data.frame(individual_id = combo_id, kring_id = kring_id, in_supercolony = in_supercolony))
    }
    # Merge with connector_summary
    connector_summary <- connector_summary %>%
      left_join(connector_bird_in_supercolony, by = "individual_id")
    cat("\nNumber of connector birds found in any supercolony (by kring mapping):", sum(connector_summary$in_supercolony, na.rm = TRUE), "\n")
  }
  
} else {
  cat("No supercolony_id data found in dyad data\n")
}

# ============================================================================
# PART 4: VISUALIZATIONS
# ============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n")

# Create output directory
dir.create("connector_analysis_output", showWarnings = FALSE)

# 1. Number of colonies vs years for connector birds
p1 <- ggplot(connector_summary, aes(x = n_colonies, y = n_years, color = total_connectivity)) +
  geom_point(alpha = 0.7, size = 3) +
  scale_color_viridis() +
  labs(title = "Connector Birds: Colonies vs Years",
       x = "Number of Colonies", y = "Number of Years",
       color = "Total Connectivity") +
  theme_minimal()

ggsave("connector_analysis_output/connector_colonies_vs_years.pdf", p1, width = 8, height = 6)
cat("Saved connector colonies vs years plot\n")

# 2. Distribution of connector birds across years
year_summary <- dyad_data %>%
  filter(bird1 %in% connector_bird_ids | bird2 %in% connector_bird_ids) %>%
  group_by(year) %>%
  summarise(
    n_connectors = n_distinct(c(bird1, bird2)),
    n_colonies = n_distinct(colony)
  )

p2 <- ggplot(year_summary, aes(x = year, y = n_connectors)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  labs(title = "Number of Connector Birds by Year",
       x = "Year", y = "Number of Connector Birds") +
  theme_minimal()

ggsave("connector_analysis_output/connector_birds_by_year.pdf", p2, width = 8, height = 6)
cat("Saved connector birds by year plot\n")

# 3. Dyad strength vs connectivity
p3 <- ggplot(connector_summary, aes(x = total_connectivity, y = avg_dyad_strength)) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Connector Birds: Connectivity vs Dyad Strength",
       x = "Total Connectivity", y = "Average Dyad Strength") +
  theme_minimal()

ggsave("connector_analysis_output/connector_connectivity_vs_strength.pdf", p3, width = 8, height = 6)
cat("Saved connector connectivity vs strength plot\n")

# ============================================================================
# PART 5: EXPORT RESULTS
# ============================================================================

cat("\n=== EXPORTING RESULTS ===\n")

# Export summary tables
write.csv(connector_summary, "connector_analysis_output/connector_birds_summary.csv", row.names = FALSE)
write.csv(year_summary, "connector_analysis_output/connector_birds_by_year.csv", row.names = FALSE)

if(exists("supercolony_connectors") && nrow(supercolony_connectors) > 0) {
  write.csv(supercolony_connectors, "connector_analysis_output/connector_birds_in_supercolonies.csv", row.names = FALSE)
}

cat("All results exported to 'connector_analysis_output/' directory\n")

# ============================================================================
# PART 6: SUMMARY
# ============================================================================

cat("\n=== ANALYSIS SUMMARY ===\n")
cat("1. Total connector birds identified:", nrow(connector_summary), "\n")
cat("2. Connector birds in supercolonies:", if(exists("supercolony_connectors")) nrow(supercolony_connectors) else 0, "\n")
cat("3. Years with connector data:", n_distinct(year_summary$year), "\n")
cat("4. Colonies with connector data:", n_distinct(dyad_data$colony[dyad_data$bird1 %in% connector_bird_ids | dyad_data$bird2 %in% connector_bird_ids]), "\n")

cat("\n=== ANALYSIS COMPLETE ===\n") 

# ============================================================================
# STEP 1: PREPARE DATA FOR CENTRALITY ANALYSIS
# ============================================================================

cat("\n=== PREPARING DATA FOR CENTRALITY ANALYSIS ===\n")

# Load all-bird degree and betweenness data
all_degree <- read.csv("network_data/degree_centrality_by_year.csv", stringsAsFactors = FALSE)
all_betweenness <- read.csv("network_data/betweenness_centrality_by_year.csv", stringsAsFactors = FALSE)

# Compute mean degree and mean betweenness for each bird
all_degree$mean_degree <- apply(all_degree[,1:5], 1, function(x) mean(as.numeric(x), na.rm=TRUE))
all_betweenness$mean_betweenness <- apply(all_betweenness[,1:5], 1, function(x) mean(as.numeric(x), na.rm=TRUE))

# Merge degree and betweenness
all_centrality <- merge(
  all_degree[,c("individual","mean_degree")],
  all_betweenness[,c("individual","mean_betweenness")],
  by="individual",
  all=TRUE
)

# Remove duplicates in connector summary (keep only one row per individual_id)
connector_summary_unique <- connector_summary %>%
  dplyr::group_by(individual_id) %>%
  dplyr::summarise(
    avg_degree = mean(avg_degree, na.rm=TRUE),
    avg_betweenness = mean(avg_betweenness, na.rm=TRUE),
    in_supercolony = any(in_supercolony, na.rm=TRUE)
  )

# Mark connector status for all birds
all_centrality$connector <- all_centrality$individual %in% connector_summary_unique$individual_id

# Add in_supercolony info for connectors
all_centrality <- all_centrality %>%
  dplyr::left_join(connector_summary_unique[,c("individual_id","in_supercolony")], by = c("individual" = "individual_id"))

cat("Centrality data prepared.\n") 

# ============================================================================
# STEP 2: CENTRALITY COMPARISON: CONNECTOR VS NON-CONNECTOR
# ============================================================================

cat("\n=== CENTRALITY COMPARISON: CONNECTOR VS NON-CONNECTOR ===\n")

library(ggplot2)
library(viridis)

# Boxplot: Degree
p_deg <- ggplot(all_centrality, aes(x = as.factor(connector), y = mean_degree, fill = as.factor(connector))) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_viridis(discrete = TRUE, option = "D") +
  labs(title = "Mean Degree: Connector vs Non-Connector Birds",
       x = "Connector Bird", y = "Mean Degree", fill = "Connector") +
  theme_minimal()
ggsave("connector_analysis_output/centrality_degree_connector_boxplot.pdf", p_deg, width = 7, height = 5)

# Boxplot: Betweenness
p_bet <- ggplot(all_centrality, aes(x = as.factor(connector), y = mean_betweenness, fill = as.factor(connector))) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_viridis(discrete = TRUE, option = "D") +
  labs(title = "Mean Betweenness: Connector vs Non-Connector Birds",
       x = "Connector Bird", y = "Mean Betweenness", fill = "Connector") +
  theme_minimal()
ggsave("connector_analysis_output/centrality_betweenness_connector_boxplot.pdf", p_bet, width = 7, height = 5)

# Statistical tests
cat("\nT-test for mean degree:\n")
deg_ttest <- t.test(mean_degree ~ connector, data = all_centrality)
print(deg_ttest)

cat("\nT-test for mean betweenness:\n")
bet_ttest <- t.test(mean_betweenness ~ connector, data = all_centrality)
print(bet_ttest)

# ============================================================================
# STEP 3: SEX-PAIR TYPE RELATIONSHIPS AND NETWORK POSITION
# ============================================================================

cat("\n=== SEX-PAIR TYPE RELATIONSHIPS AND NETWORK POSITION ===\n")

# Load integrated dyad data
integrated_dyad <- read.csv("dyad_full_integrated.csv", stringsAsFactors = FALSE)

# Only keep relevant columns
integrated_dyad <- integrated_dyad[,c("bird1","bird2","sex_pair")]

# Get unique connector bird IDs
connector_ids <- unique(connector_summary$individual_id)

# For each connector bird, count dyad types
sexpair_counts <- data.frame()
for (bird in connector_ids) {
  dyads <- integrated_dyad[integrated_dyad$bird1 == bird | integrated_dyad$bird2 == bird,]
  n_FF <- sum(dyads$sex_pair == "FF", na.rm=TRUE)
  n_FM <- sum(dyads$sex_pair == "FM", na.rm=TRUE)
  n_MF <- sum(dyads$sex_pair == "MF", na.rm=TRUE)
  n_MM <- sum(dyads$sex_pair == "MM", na.rm=TRUE)
  sexpair_counts <- rbind(sexpair_counts, data.frame(individual_id=bird, n_FF=n_FF, n_FM=n_FM, n_MF=n_MF, n_MM=n_MM))
}

# Merge with connector centrality
connector_sexpair_centrality <- connector_summary_unique %>%
  dplyr::left_join(sexpair_counts, by="individual_id")

# Remove rows with NA or non-finite values for plotting/correlation
plot_data <- connector_sexpair_centrality %>%
  dplyr::filter(!is.na(n_FF), !is.na(avg_degree), !is.na(avg_betweenness), is.finite(n_FF), is.finite(avg_degree), is.finite(avg_betweenness))

cat("\nSummary of data used for FF dyad vs. centrality analysis:\n")
print(summary(plot_data[, c("n_FF", "avg_degree", "avg_betweenness")]))

# Only run correlation if there is variance
if (sd(plot_data$n_FF) > 0 & sd(plot_data$avg_degree) > 0) {
  cat("\nCorrelation between number of FF dyads and average degree:\n")
  print(cor.test(plot_data$n_FF, plot_data$avg_degree))
} else {
  cat("\nNot enough variance in n_FF or avg_degree for correlation.\n")
}

if (sd(plot_data$n_FF) > 0 & sd(plot_data$avg_betweenness) > 0) {
  cat("\nCorrelation between number of FF dyads and average betweenness:\n")
  print(cor.test(plot_data$n_FF, plot_data$avg_betweenness))
} else {
  cat("\nNot enough variance in n_FF or avg_betweenness for correlation.\n")
}

# Visualize only if there is data
if (nrow(plot_data) > 0) {
  p_ff_deg <- ggplot(plot_data, aes(x=n_FF, y=avg_degree)) +
    geom_point(alpha=0.7) +
    geom_smooth(method="lm", se=TRUE) +
    labs(title="Connector Birds: Number of FF Dyads vs. Average Degree",
         x="Number of FF Dyads", y="Average Degree") +
    theme_minimal()
  ggsave("connector_analysis_output/connector_ffdyads_vs_degree.pdf", p_ff_deg, width=7, height=5)

  p_ff_bet <- ggplot(plot_data, aes(x=n_FF, y=avg_betweenness)) +
    geom_point(alpha=0.7) +
    geom_smooth(method="lm", se=TRUE) +
    labs(title="Connector Birds: Number of FF Dyads vs. Average Betweenness",
         x="Number of FF Dyads", y="Average Betweenness") +
    theme_minimal()
  ggsave("connector_analysis_output/connector_ffdyads_vs_betweenness.pdf", p_ff_bet, width=7, height=5)
} else {
  cat("\nNo data available for plotting FF dyads vs. centrality.\n")
}
cat("\nSex-pair type analysis complete.\n")

# ============================================================================
# STEP 4: SUMMARY OF SEX-PAIR TYPES AND NETWORK METRICS BY COLONY/SUPERCOLONY
# ============================================================================

cat("\n=== SUMMARY: SEX-PAIR TYPES AND NETWORK METRICS BY COLONY/SUPERCOLONY ===\n")

# Always reload the full integrated dyad data for colony/supercolony summaries
integrated_dyad <- read.csv("dyad_full_integrated.csv", stringsAsFactors = FALSE)

# --- Sex-pair type summary by colony ---
colony_col <- if ("colony" %in% names(integrated_dyad)) {
  "colony"
} else if ("colony_clean" %in% names(integrated_dyad)) {
  "colony_clean"
} else {
  stop("No colony column found in dyad data.")
}

sexpair_by_colony <- integrated_dyad %>%
  dplyr::group_by(.data[[colony_col]], sex_pair) %>%
  dplyr::summarise(n_dyads = dplyr::n()) %>%
  tidyr::pivot_wider(names_from = sex_pair, values_from = n_dyads, values_fill = 0)
cat("\nSex-pair type counts by colony:\n")
print(sexpair_by_colony)

# --- Sex-pair type summary by supercolony (if available) ---
if ("supercolony_id" %in% names(integrated_dyad)) {
  sexpair_by_supercolony <- integrated_dyad %>%
    dplyr::filter(!is.na(supercolony_id) & supercolony_id != "") %>%
    dplyr::group_by(supercolony_id, sex_pair) %>%
    dplyr::summarise(n_dyads = dplyr::n()) %>%
    tidyr::pivot_wider(names_from = sex_pair, values_from = n_dyads, values_fill = 0)
  cat("\nSex-pair type counts by supercolony:\n")
  print(sexpair_by_supercolony)
} else {
  cat("\nNo supercolony_id column found in dyad data.\n")
}

# --- Network metrics by colony ---
metrics_by_colony <- integrated_dyad %>%
  dplyr::group_by(.data[[colony_col]]) %>%
  dplyr::summarise(
    mean_degree = mean(rowMeans(cbind(degree_bird1, degree_bird2), na.rm=TRUE), na.rm=TRUE),
    mean_betweenness = mean(rowMeans(cbind(betweenness_bird1, betweenness_bird2), na.rm=TRUE), na.rm=TRUE),
    mean_clustering = if ("clustering" %in% names(integrated_dyad)) mean(clustering, na.rm=TRUE) else NA
  )
cat("\nNetwork metrics by colony:\n")
print(metrics_by_colony)

# --- Network metrics by supercolony (if available) ---
if ("supercolony_id" %in% names(integrated_dyad)) {
  metrics_by_supercolony <- integrated_dyad %>%
    dplyr::filter(!is.na(supercolony_id) & supercolony_id != "") %>%
    dplyr::group_by(supercolony_id) %>%
    dplyr::summarise(
      mean_degree = mean(rowMeans(cbind(degree_bird1, degree_bird2), na.rm=TRUE), na.rm=TRUE),
      mean_betweenness = mean(rowMeans(cbind(betweenness_bird1, betweenness_bird2), na.rm=TRUE), na.rm=TRUE),
      mean_clustering = if ("clustering" %in% names(integrated_dyad)) mean(clustering, na.rm=TRUE) else NA
    )
  cat("\nNetwork metrics by supercolony:\n")
  print(metrics_by_supercolony)
} else {
  cat("\nNo supercolony_id column found in dyad data.\n")
}

# --- Statistical comparison: colonies vs supercolonies ---
cat("\n=== STATISTICAL COMPARISON: COLONY VS SUPERCOLONY ===\n")

# Prepare data for comparison
colony_metrics <- metrics_by_colony %>%
  dplyr::mutate(type = "colony") %>%
  dplyr::rename(id = !!colony_col)
supercolony_metrics <- if (exists("metrics_by_supercolony")) {
  metrics_by_supercolony %>%
    dplyr::mutate(type = "supercolony") %>%
    dplyr::rename(id = supercolony_id)
} else {
  NULL
}
combined_metrics <- dplyr::bind_rows(colony_metrics, supercolony_metrics)

# Remove rows with NA/NaN for tests
deg_data <- combined_metrics %>% dplyr::filter(!is.na(mean_degree), !is.nan(mean_degree))
bet_data <- combined_metrics %>% dplyr::filter(!is.na(mean_betweenness), !is.nan(mean_betweenness))

# T-test and Wilcoxon for mean degree
cat("\nT-test for mean degree (colony vs supercolony):\n")
if (length(unique(deg_data$type)) == 2) {
  print(t.test(mean_degree ~ type, data = deg_data))
  cat("\nWilcoxon test for mean degree (colony vs supercolony):\n")
  print(wilcox.test(mean_degree ~ type, data = deg_data))
} else {
  cat("Not enough groups for comparison.\n")
}

# T-test and Wilcoxon for mean betweenness
cat("\nT-test for mean betweenness (colony vs supercolony):\n")
if (length(unique(bet_data$type)) == 2) {
  print(t.test(mean_betweenness ~ type, data = bet_data))
  cat("\nWilcoxon test for mean betweenness (colony vs supercolony):\n")
  print(wilcox.test(mean_betweenness ~ type, data = bet_data))
} else {
  cat("Not enough groups for comparison.\n")
}

# Output combined summary table
cat("\nCombined summary table (colony and supercolony):\n")
print(combined_metrics)

cat("\nColony/supercolony summary complete.\n")

# ============================================================================
# STEP 5: NEXT STEPS
# ============================================================================

cat("\n=== NEXT STEPS ===\n")
cat("1. For each connector bird, summarize the types of dyads (FF, FM, MF, MM) they participate in, and relate to centrality.\n")
cat("2. Proceed to code this step next.\n") 

# ============================================================================
# STEP 6: BRIDGE BIRD ANALYSIS FOR SPECIFIC COLONIES
# ============================================================================

cat("\n=== BRIDGE BIRD ANALYSIS FOR SPECIFIC COLONIES ===\n")

library(readxl)
kring_map <- read_excel("../data/index/identification.xlsx")
kring_map <- kring_map %>% dplyr::select(Metal, Combo)

# List of colonies to analyze
target_colonies <- c("MSTO_29", "LLOD_17", "SPRA_03")

# Detect correct colony column
dyad_colnames <- names(integrated_dyad)
colony_col <- if ("colony" %in% dyad_colnames) {
  "colony"
} else if ("colony_clean" %in% dyad_colnames) {
  "colony_clean"
} else {
  stop("No colony column found in dyad data.")
}

for (col in target_colonies) {
  cat("\n--- Colony:", col, "---\n")
  dyad_col <- integrated_dyad[integrated_dyad$colony == col, ]
  # Dynamically build bridge_map for this colony: only birds present in bird1 or bird2
  birds_in_colony <- unique(c(dyad_col$bird1, dyad_col$bird2))
  bridge_map_col <- kring_map %>% dplyr::filter(Combo %in% birds_in_colony)
  bridge_map_col <- bridge_map_col[!duplicated(bridge_map_col$Combo), ]
  colony_counts <- data.frame()
  for (i in 1:nrow(bridge_map_col)) {
    combo_id <- bridge_map_col$Combo[i]
    dyads <- dyad_col[dyad_col$bird1 == combo_id | dyad_col$bird2 == combo_id, ]
    n_dyads <- nrow(dyads)
    sexpair_vals <- as.character(dyads[[sexpair_col]])
    n_FF <- sum(tolower(sexpair_vals) == "ff", na.rm=TRUE)
    n_FM <- sum(tolower(sexpair_vals) == "fm", na.rm=TRUE)
    n_MF <- sum(tolower(sexpair_vals) == "mf", na.rm=TRUE)
    n_MM <- sum(tolower(sexpair_vals) == "mm", na.rm=TRUE)
    # Infer sex from bird1_sex and bird2_sex columns
    sex_vals <- c()
    if ("bird1_sex" %in% names(dyads)) {
      sex_vals <- c(sex_vals, dyads$bird1_sex[dyads$bird1 == combo_id])
    }
    if ("bird2_sex" %in% names(dyads)) {
      sex_vals <- c(sex_vals, dyads$bird2_sex[dyads$bird2 == combo_id])
    }
    sex_vals <- unique(na.omit(sex_vals))
    if (length(sex_vals) == 1) {
      sex <- sex_vals[1]
    } else if (length(sex_vals) > 1) {
      sex <- "ambiguous"
    } else {
      sex <- NA
    }
    # Get mean_degree and mean_betweenness for this bird (from previous summary if available)
    mean_degree <- NA
    mean_betweenness <- NA
    if (combo_id %in% bridge_bird_summaries$kring_id) {
      mean_degree <- bridge_bird_summaries$mean_degree[bridge_bird_summaries$kring_id == combo_id][1]
      mean_betweenness <- bridge_bird_summaries$mean_betweenness[bridge_bird_summaries$kring_id == combo_id][1]
    }
    colony_counts <- rbind(colony_counts, data.frame(
      combo_id = combo_id,
      metal_id = bridge_map_col$Metal[i],
      sex = sex,
      mean_degree = mean_degree,
      mean_betweenness = mean_betweenness,
      n_FF = n_FF, n_FM = n_FM, n_MF = n_MF, n_MM = n_MM
    ))
  }
  # Deduplicate output table by combo_id
  colony_counts <- colony_counts[!duplicated(colony_counts$combo_id), ]
  # Add summary row
  summary_row <- data.frame(
    combo_id = "SUMMARY",
    metal_id = NA,
    sex = NA,
    mean_degree = mean(colony_counts$mean_degree, na.rm=TRUE),
    mean_betweenness = mean(colony_counts$mean_betweenness, na.rm=TRUE),
    n_FF = sum(colony_counts$n_FF, na.rm=TRUE),
    n_FM = sum(colony_counts$n_FM, na.rm=TRUE),
    n_MF = sum(colony_counts$n_MF, na.rm=TRUE),
    n_MM = sum(colony_counts$n_MM, na.rm=TRUE)
  )
  colony_counts <- rbind(colony_counts, summary_row)
  print(colony_counts)
}
cat("\nBridge bird analysis complete.\n") 

# =====================
# STEP 7: DYADIC CONTEXT FOR BRIDGE/CONNECTOR BIRDS (ID TRANSLATION FIX)
# =====================

cat("\n=== DYADIC CONTEXT FOR BRIDGE/CONNECTOR BIRDS (ID TRANSLATION FIX) ===\n")

# Read kring mapping (Combo = 4-letter, Metal = metal ring)
library(readxl)
kring_map <- read_excel("../data/index/identification.xlsx")
kring_map <- kring_map %>% dplyr::select(Metal, Combo)

# Combine all bridge birds from all target colonies into one vector (unique)
all_bridge_birds <- unique(bridge_bird_summaries$kring_id)

# Robustly detect the correct dyad type column (case-insensitive, prefer 'sex_pair')
dyad_colnames <- names(integrated_dyad)
sexpair_col <- NULL
if ("sex_pair" %in% dyad_colnames) {
  sexpair_col <- "sex_pair"
} else if ("sex_pair.x" %in% dyad_colnames) {
  sexpair_col <- "sex_pair.x"
} else if ("sex_pair.y" %in% dyad_colnames) {
  sexpair_col <- "sex_pair.y"
} else if (any(tolower(dyad_colnames) == "sex_pair")) {
  sexpair_col <- dyad_colnames[tolower(dyad_colnames) == "sex_pair"][1]
} else {
  stop("No sex_pair column found in integrated_dyad.")
}

# Build union of all birds present in the target colonies
dyad_birds_in_targets <- unique(unlist(
  lapply(target_colonies, function(col) {
    dyad_col <- integrated_dyad[integrated_dyad$colony == col, ]
    unique(c(dyad_col$bird1, dyad_col$bird2))
  })
))
# Filter kring_map to only those birds
bridge_map <- kring_map %>% dplyr::filter(Combo %in% dyad_birds_in_targets)
bridge_map <- bridge_map[!duplicated(bridge_map$Combo), ]

# For each bridge/connector bird, count dyad sex-pair types in integrated_dyad using Combo (4-letter code)
all_dyad_counts <- data.frame()
for (i in 1:nrow(bridge_map)) {
  combo_id <- bridge_map$Combo[i]
  dyads <- integrated_dyad[integrated_dyad$bird1 == combo_id | integrated_dyad$bird2 == combo_id,]
  n_dyads <- nrow(dyads)
  sexpair_vals <- as.character(dyads[[sexpair_col]])
  n_FF <- sum(tolower(sexpair_vals) == "ff", na.rm=TRUE)
  n_FM <- sum(tolower(sexpair_vals) == "fm", na.rm=TRUE)
  n_MF <- sum(tolower(sexpair_vals) == "mf", na.rm=TRUE)
  n_MM <- sum(tolower(sexpair_vals) == "mm", na.rm=TRUE)
  # Infer sex from bird1_sex and bird2_sex columns
  sex_vals <- c()
  if ("bird1_sex" %in% names(dyads)) {
    sex_vals <- c(sex_vals, dyads$bird1_sex[dyads$bird1 == combo_id])
  }
  if ("bird2_sex" %in% names(dyads)) {
    sex_vals <- c(sex_vals, dyads$bird2_sex[dyads$bird2 == combo_id])
  }
  sex_vals <- unique(na.omit(sex_vals))
  if (length(sex_vals) == 1) {
    sex <- sex_vals[1]
  } else if (length(sex_vals) > 1) {
    sex <- "ambiguous"
  } else {
    sex <- NA
  }
  # Get mean_degree and mean_betweenness for this bird (from previous summary if available)
  mean_degree <- NA
  mean_betweenness <- NA
  if (combo_id %in% bridge_bird_summaries$kring_id) {
    mean_degree <- bridge_bird_summaries$mean_degree[bridge_bird_summaries$kring_id == combo_id][1]
    mean_betweenness <- bridge_bird_summaries$mean_betweenness[bridge_bird_summaries$kring_id == combo_id][1]
  }
  all_dyad_counts <- rbind(all_dyad_counts, data.frame(
    combo_id = combo_id,
    metal_id = bridge_map$Metal[i],
    sex = sex,
    mean_degree = mean_degree,
    mean_betweenness = mean_betweenness,
    n_FF = n_FF, n_FM = n_FM, n_MF = n_MF, n_MM = n_MM
  ))
}
# Deduplicate output table by combo_id
all_dyad_counts <- all_dyad_counts[!duplicated(all_dyad_counts$combo_id), ]

cat("\nBridge/connector bird dyadic context table (robust):\n")
print(all_dyad_counts)

# Add summary row (totals and means)
summary_row <- data.frame(
  combo_id = "SUMMARY",
  metal_id = NA,
  sex = NA,
  mean_degree = mean(all_dyad_counts$mean_degree, na.rm=TRUE),
  mean_betweenness = mean(all_dyad_counts$mean_betweenness, na.rm=TRUE),
  n_FF = sum(all_dyad_counts$n_FF, na.rm=TRUE),
  n_FM = sum(all_dyad_counts$n_FM, na.rm=TRUE),
  n_MF = sum(all_dyad_counts$n_MF, na.rm=TRUE),
  n_MM = sum(all_dyad_counts$n_MM, na.rm=TRUE)
)
all_dyad_counts <- rbind(all_dyad_counts, summary_row)

cat("\nBridge/connector bird dyadic context table with summary (robust):\n")
print(all_dyad_counts) 

# =====================
# STEP 8: PER-COLONY SUMMARIES FOR BRIDGE/CONNECTOR BIRDS
# =====================

cat("\n=== PER-COLONY SUMMARIES FOR BRIDGE/CONNECTOR BIRDS ===\n")

# Get unique colonies from dyad data
unique_colonies <- sort(unique(integrated_dyad$colony))

# For each colony, compute bridge/connector bird dyadic context table
per_colony_summaries <- list()
for (col in unique_colonies) {
  cat("\n--- Colony:", col, "---\n")
  # Subset dyad data for this colony
  dyad_col <- integrated_dyad[integrated_dyad$colony == col, ]
  # For each bridge/connector bird, count dyads in this colony
  colony_counts <- data.frame()
  for (i in 1:nrow(bridge_map)) {
    combo_id <- bridge_map$Combo[i]
    dyads <- dyad_col[dyad_col$bird1 == combo_id | dyad_col$bird2 == combo_id, ]
    n_dyads <- nrow(dyads)
    sexpair_vals <- as.character(dyads[[sexpair_col]])
    n_FF <- sum(tolower(sexpair_vals) == "ff", na.rm=TRUE)
    n_FM <- sum(tolower(sexpair_vals) == "fm", na.rm=TRUE)
    n_MF <- sum(tolower(sexpair_vals) == "mf", na.rm=TRUE)
    n_MM <- sum(tolower(sexpair_vals) == "mm", na.rm=TRUE)
    # Infer sex from bird1_sex and bird2_sex columns
    sex_vals <- c()
    if ("bird1_sex" %in% names(dyads)) {
      sex_vals <- c(sex_vals, dyads$bird1_sex[dyads$bird1 == combo_id])
    }
    if ("bird2_sex" %in% names(dyads)) {
      sex_vals <- c(sex_vals, dyads$bird2_sex[dyads$bird2 == combo_id])
    }
    sex_vals <- unique(na.omit(sex_vals))
    if (length(sex_vals) == 1) {
      sex <- sex_vals[1]
    } else if (length(sex_vals) > 1) {
      sex <- "ambiguous"
    } else {
      sex <- NA
    }
    # Get mean_degree and mean_betweenness for this bird (from previous summary if available)
    mean_degree <- NA
    mean_betweenness <- NA
    if (combo_id %in% bridge_bird_summaries$kring_id) {
      mean_degree <- bridge_bird_summaries$mean_degree[bridge_bird_summaries$kring_id == combo_id][1]
      mean_betweenness <- bridge_bird_summaries$mean_betweenness[bridge_bird_summaries$kring_id == combo_id][1]
    }
    colony_counts <- rbind(colony_counts, data.frame(
      combo_id = combo_id,
      metal_id = bridge_map$Metal[i],
      sex = sex,
      mean_degree = mean_degree,
      mean_betweenness = mean_betweenness,
      n_FF = n_FF, n_FM = n_FM, n_MF = n_MF, n_MM = n_MM
    ))
  }
  # Deduplicate output table by combo_id
  colony_counts <- colony_counts[!duplicated(colony_counts$combo_id), ]
  # Add summary row
  summary_row <- data.frame(
    combo_id = "SUMMARY",
    metal_id = NA,
    sex = NA,
    mean_degree = mean(colony_counts$mean_degree, na.rm=TRUE),
    mean_betweenness = mean(colony_counts$mean_betweenness, na.rm=TRUE),
    n_FF = sum(colony_counts$n_FF, na.rm=TRUE),
    n_FM = sum(colony_counts$n_FM, na.rm=TRUE),
    n_MF = sum(colony_counts$n_MF, na.rm=TRUE),
    n_MM = sum(colony_counts$n_MM, na.rm=TRUE)
  )
  colony_counts <- rbind(colony_counts, summary_row)
  print(colony_counts)
  per_colony_summaries[[col]] <- colony_counts
}

cat("\n=== SUMMARY ACROSS ALL COLONIES ===\n")
# Optionally, combine all per-colony summaries for further analysis
# (e.g., rbind all summary rows)
colony_summary_table <- do.call(rbind, lapply(names(per_colony_summaries), function(col) {
  row <- per_colony_summaries[[col]]
  row_summary <- row[row$combo_id == "SUMMARY", ]
  row_summary$colony <- col
  row_summary
}))
print(colony_summary_table) 