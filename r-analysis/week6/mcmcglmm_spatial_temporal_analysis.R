# mcmcglmm_spatial_temporal_analysis.R
# Purpose: Comprehensive MCMCglmm analysis of weaver bird social dynamics
# Including social association drivers, temporal autocorrelation, environmental stressors, and state transitions

library(MCMCglmm)
library(dplyr)
library(ggplot2)
library(tidyr)
library(corrplot)
library(gridExtra)
library(knitr)
library(coda)
library(readr)

setwd("~/Code/Birds")

# 1. Load and prepare data
cat("Loading cleaned spatial-temporal dyad data...\n")
data <- read_csv("week6/cleaned_dyad_year_spatial2.csv")

# Check data structure
cat("Data dimensions:", dim(data), "\n")
cat("Years:", sort(unique(data$year)), "\n")
cat("Unique dyads:", length(unique(data$dyad_id)), "\n")

# 2. Data preprocessing for MCMCglmm
# Standardize continuous predictors for better convergence
data <- data %>%
  mutate(
    # Standardize continuous variables
    association_std = scale(association)[,1],
    dist_m_std = scale(dist_m)[,1],
    r_std = scale(r)[,1],
    pair_bond_std = scale(pair_bond_strength)[,1],
    association_lag_std = scale(association_lag)[,1],
    
    # Create binary variables for state transitions
    high_association = ifelse(association > median(association, na.rm=TRUE), 1, 0),
    has_pair_bond = ifelse(pair_bond_strength > 0, 1, 0),
    
    # Create factor variables
    year_f = factor(year),
    sex_combo_f = factor(sex_combo),
    plot_f = factor(plot)
  )

# Remove any remaining NAs
data_clean <- data %>% filter(!is.na(association_std) & !is.na(dist_m_std) & 
                               !is.na(r_std) & !is.na(pair_bond_std))

cat("Clean data dimensions:", dim(data_clean), "\n")

# Additional cleaning: remove any rows with NA in fixed predictors
data_clean <- data_clean %>%
  filter(!is.na(dist_m_std) & !is.na(r_std) & !is.na(pair_bond_std) & 
         !is.na(sex_combo_f) & !is.na(association_std))

cat("After removing NA in fixed predictors:", dim(data_clean), "\n")

# Check for any remaining NAs
na_check <- data_clean %>%
  select(association_std, dist_m_std, r_std, pair_bond_std, sex_combo_f) %>%
  summarise_all(~sum(is.na(.)))

cat("NA counts in key variables:\n")
print(na_check)

# Ensure sex_combo_f has no empty levels
data_clean$sex_combo_f <- droplevels(data_clean$sex_combo_f)
cat("Sex combination levels:", levels(data_clean$sex_combo_f), "\n")

# Final data check before modeling
cat("Final data summary:\n")
cat("Total rows:", nrow(data_clean), "\n")
cat("Unique dyads:", length(unique(data_clean$dyad_id)), "\n")
cat("Unique individuals:", length(unique(c(data_clean$id1, data_clean$id2))), "\n")

# Check for any infinite values
inf_check <- data_clean %>%
  select(association_std, dist_m_std, r_std, pair_bond_std) %>%
  summarise_all(~sum(is.infinite(.)))

cat("Infinite values in key variables:\n")
print(inf_check)

# Remove any infinite values
data_clean <- data_clean %>%
  filter(!is.infinite(association_std) & !is.infinite(dist_m_std) & 
         !is.infinite(r_std) & !is.infinite(pair_bond_std))

cat("After removing infinite values:", nrow(data_clean), "rows\n")

# Ensure all variables are numeric/factor as expected
cat("Variable types:\n")
cat("association_std:", class(data_clean$association_std), "\n")
cat("dist_m_std:", class(data_clean$dist_m_std), "\n")
cat("r_std:", class(data_clean$r_std), "\n")
cat("pair_bond_std:", class(data_clean$pair_bond_std), "\n")
cat("sex_combo_f:", class(data_clean$sex_combo_f), "\n")

# 3. Set up MCMCglmm parameters - OPTIMIZED FOR SPEED
# Reduced iterations for faster computation while maintaining quality
nitt <- 13000   # Reduced from 13000
burnin <- 3000 # Reduced from 3000  
thin <- 10     # Increased from 10 (less frequent sampling)

# Function to create flexible priors based on number of random effects
create_gaussian_prior <- function(n_random_effects) {
  G_list <- list()
  for(i in 1:n_random_effects) {
    G_list[[paste0("G", i)]] <- list(V = 1, nu = 0.002)
  }
  
  return(list(
    R = list(V = 1, nu = 0.002),
    G = G_list
  ))
}

create_binomial_prior <- function(n_random_effects) {
  G_list <- list()
  for(i in 1:n_random_effects) {
    G_list[[paste0("G", i)]] <- list(V = 1, nu = 0.002)
  }
  
  return(list(
    R = list(V = 1, fix = 1),
    G = G_list
  ))
}

# Check that priors are properly defined
cat("Prior functions created successfully\n")

# Enable parallel processing if available
if(require(parallel) && require(doParallel)) {
  cat("Parallel processing enabled\n")
  num_cores <- min(2, detectCores() - 1)  # Use 2 cores max for MCMCglmm
  registerDoParallel(cores = num_cores)
} else {
  cat("Running in serial mode\n")
}

# 4. MODEL 1: Simple Association Drivers - OPTIMIZED
cat("\n=== MODEL 1: Simple Association Drivers ===\n")
cat("Predicting association strength from core social and spatial factors\n")
cat("Using optimized settings: nitt=", nitt, ", burnin=", burnin, ", thin=", thin, "\n")

# Create prior for Model 1 (3 random effects: dyad_id, id1, id2)
prior_model1 <- create_gaussian_prior(3)

# Run multiple chains for proper diagnostics
set.seed(123)
cat("Running Chain 1...\n")
tryCatch({
  model1_chain1 <- MCMCglmm(
    association_std ~ dist_m_std + r_std + pair_bond_std + sex_combo_f,
    random = ~ dyad_id + id1 + id2,
    data = data_clean,
    family = "gaussian",
    prior = prior_model1,
    nitt = nitt,
    burnin = burnin,
    thin = thin,
    verbose = FALSE,
    pr = TRUE  # Store predictions for diagnostics
  )
  cat("Chain 1 completed successfully\n")
}, error = function(e) {
  cat("Error in Chain 1:", e$message, "\n")
  cat("Data summary before error:\n")
  print(summary(data_clean[, c("association_std", "dist_m_std", "r_std", "pair_bond_std", "sex_combo_f")]))
  stop(e)
})

set.seed(456)
cat("Running Chain 2...\n")
tryCatch({
  model1_chain2 <- MCMCglmm(
    association_std ~ dist_m_std + r_std + pair_bond_std + sex_combo_f,
    random = ~ dyad_id + id1 + id2,
    data = data_clean,
    family = "gaussian",
    prior = prior_model1,
    nitt = nitt,
    burnin = burnin,
    thin = thin,
    verbose = FALSE,
    pr = TRUE
  )
  cat("Chain 2 completed successfully\n")
}, error = function(e) {
  cat("Error in Chain 2:", e$message, "\n")
  stop(e)
})

# Use first chain for main results
model1 <- model1_chain1

# Quick diagnostics (faster than full diagnostics)
cat("Model 1 Quick Diagnostics:\n")
cat("Effective Sample Size (min):", min(effectiveSize(model1$Sol)), "\n")
cat("DIC:", model1$DIC, "\n")

# Check Gelman-Rubin with multiple chains
tryCatch({
  # Combine chains for Gelman-Rubin
  combined_chains <- mcmc.list(
    mcmc(model1_chain1$Sol),
    mcmc(model1_chain2$Sol)
  )
  gelman_result <- gelman.diag(combined_chains)
  cat("Max Gelman-Rubin:", max(gelman_result$psrf[,1]), "\n")
  if(max(gelman_result$psrf[,1]) < 1.1) {
    cat("✓ Chains converged well\n")
  } else {
    cat("⚠ Chains may need more iterations\n")
  }
}, error = function(e) {
  cat("Gelman-Rubin diagnostic failed:", e$message, "\n")
})

# Model 1 summary
summary_model1 <- summary(model1)
print(summary_model1)

# 5. MODEL 2: Add Environmental Stressors - OPTIMIZED
cat("\n=== MODEL 2: Add Environmental Stressors ===\n")
cat("Adding disturbance effects to association model\n")

# Create prior for Model 2 (5 random effects: dyad_id, id1, id2, colony1, colony2)
prior_model2 <- create_gaussian_prior(5)

set.seed(123)
cat("Running Chain 1...\n")
model2_chain1 <- MCMCglmm(
  association_std ~ dist_m_std + r_std + pair_bond_std + total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_clean,
  family = "gaussian",
  prior = prior_model2,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

set.seed(456)
cat("Running Chain 2...\n")
model2_chain2 <- MCMCglmm(
  association_std ~ dist_m_std + r_std + pair_bond_std + total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_clean,
  family = "gaussian",
  prior = prior_model2,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

model2 <- model2_chain1

# Model 2 diagnostics and summary
cat("Model 2 Quick Diagnostics:\n")
cat("Effective Sample Size (min):", min(effectiveSize(model2$Sol)), "\n")
cat("DIC:", model2$DIC, "\n")

tryCatch({
  combined_chains <- mcmc.list(
    mcmc(model2_chain1$Sol),
    mcmc(model2_chain2$Sol)
  )
  gelman_result <- gelman.diag(combined_chains)
  cat("Max Gelman-Rubin:", max(gelman_result$psrf[,1]), "\n")
}, error = function(e) {
  cat("Gelman-Rubin diagnostic failed:", e$message, "\n")
})

summary_model2 <- summary(model2)
print(summary_model2)

# 6. MODEL 3: Temporal Autocorrelation - OPTIMIZED
cat("\n=== MODEL 3: Temporal Autocorrelation ===\n")
cat("Adding lagged association to model temporal stability\n")

# Filter for dyads with lagged data
data_with_lag <- data_clean %>% filter(!is.na(association_lag_std))

cat("Data with lagged association:", dim(data_with_lag), "\n")

# Create prior for Model 3 (5 random effects: dyad_id, id1, id2, colony1, colony2)
prior_model3 <- create_gaussian_prior(5)

set.seed(123)
cat("Running Chain 1...\n")
model3_chain1 <- MCMCglmm(
  association_std ~ association_lag_std + dist_m_std + r_std + pair_bond_std + 
                   total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_with_lag,
  family = "gaussian",
  prior = prior_model3,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

set.seed(456)
cat("Running Chain 2...\n")
model3_chain2 <- MCMCglmm(
  association_std ~ association_lag_std + dist_m_std + r_std + pair_bond_std + 
                   total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_with_lag,
  family = "gaussian",
  prior = prior_model3,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

model3 <- model3_chain1

# Model 3 diagnostics and summary
cat("Model 3 Quick Diagnostics:\n")
cat("Effective Sample Size (min):", min(effectiveSize(model3$Sol)), "\n")
cat("DIC:", model3$DIC, "\n")

tryCatch({
  combined_chains <- mcmc.list(
    mcmc(model3_chain1$Sol),
    mcmc(model3_chain2$Sol)
  )
  gelman_result <- gelman.diag(combined_chains)
  cat("Max Gelman-Rubin:", max(gelman_result$psrf[,1]), "\n")
}, error = function(e) {
  cat("Gelman-Rubin diagnostic failed:", e$message, "\n")
})

summary_model3 <- summary(model3)
print(summary_model3)

# 7. MODEL 4: State Transitions (Binary) - OPTIMIZED
cat("\n=== MODEL 4: State Transitions ===\n")
cat("Modeling probability of high association state\n")

# Create prior for Model 4 (5 random effects: dyad_id, id1, id2, colony1, colony2)
prior_model4 <- create_binomial_prior(5)

set.seed(123)
cat("Running Chain 1...\n")
model4_chain1 <- MCMCglmm(
  high_association ~ dist_m_std + r_std + pair_bond_std + total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_clean,
  family = "categorical",
  prior = prior_model4,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

set.seed(456)
cat("Running Chain 2...\n")
model4_chain2 <- MCMCglmm(
  high_association ~ dist_m_std + r_std + pair_bond_std + total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_clean,
  family = "categorical",
  prior = prior_model4,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

model4 <- model4_chain1

# Model 4 diagnostics and summary
cat("Model 4 Quick Diagnostics:\n")
cat("Effective Sample Size (min):", min(effectiveSize(model4$Sol)), "\n")
cat("DIC:", model4$DIC, "\n")

tryCatch({
  combined_chains <- mcmc.list(
    mcmc(model4_chain1$Sol),
    mcmc(model4_chain2$Sol)
  )
  gelman_result <- gelman.diag(combined_chains)
  cat("Max Gelman-Rubin:", max(gelman_result$psrf[,1]), "\n")
}, error = function(e) {
  cat("Gelman-Rubin diagnostic failed:", e$message, "\n")
})

summary_model4 <- summary(model4)
print(summary_model4)

# 8. MODEL 5: Pair Bond Transitions - OPTIMIZED
cat("\n=== MODEL 5: Pair Bond Transitions ===\n")
cat("Modeling probability of pair bond formation\n")

# Create prior for Model 5 (5 random effects: dyad_id, id1, id2, colony1, colony2)
prior_model5 <- create_binomial_prior(5)

set.seed(123)
cat("Running Chain 1...\n")
model5_chain1 <- MCMCglmm(
  has_pair_bond ~ dist_m_std + r_std + total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_clean,
  family = "categorical",
  prior = prior_model5,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

set.seed(456)
cat("Running Chain 2...\n")
model5_chain2 <- MCMCglmm(
  has_pair_bond ~ dist_m_std + r_std + total_disturbances + sex_combo_f,
  random = ~ dyad_id + id1 + id2 + colony1 + colony2,
  data = data_clean,
  family = "categorical",
  prior = prior_model5,
  nitt = nitt,
  burnin = burnin,
  thin = thin,
  verbose = FALSE,
  pr = TRUE
)

model5 <- model5_chain1

# Model 5 diagnostics and summary
cat("Model 5 Quick Diagnostics:\n")
cat("Effective Sample Size (min):", min(effectiveSize(model5$Sol)), "\n")
cat("DIC:", model5$DIC, "\n")

tryCatch({
  combined_chains <- mcmc.list(
    mcmc(model5_chain1$Sol),
    mcmc(model5_chain2$Sol)
  )
  gelman_result <- gelman.diag(combined_chains)
  cat("Max Gelman-Rubin:", max(gelman_result$psrf[,1]), "\n")
}, error = function(e) {
  cat("Gelman-Rubin diagnostic failed:", e$message, "\n")
})

summary_model5 <- summary(model5)
print(summary_model5)

# 9. Model Comparison
cat("\n=== MODEL COMPARISON ===\n")

# Calculate DIC for comparison
dics <- data.frame(
  Model = c("Simple Drivers", "Environmental Stressors", "Temporal Autocorrelation", 
            "State Transitions", "Pair Bond Transitions"),
  DIC = c(model1$DIC, model2$DIC, model3$DIC, model4$DIC, model5$DIC),
  Observations = c(nrow(data_clean), nrow(data_clean), nrow(data_with_lag), 
                   nrow(data_clean), nrow(data_clean))
)

print(dics)

# 10. Visualization Functions
create_effect_plot <- function(model, title) {
  # Extract fixed effects
  fixed_effects <- summary(model)$solutions
  
  # Create data frame for plotting
  plot_data <- data.frame(
    Parameter = rownames(fixed_effects),
    Estimate = fixed_effects[, "post.mean"],
    Lower = fixed_effects[, "l-95% CI"],
    Upper = fixed_effects[, "u-95% CI"],
    Significant = ifelse(fixed_effects[, "pMCMC"] < 0.05, "Yes", "No")
  )
  
  # Remove intercept for cleaner plot
  plot_data <- plot_data[plot_data$Parameter != "(Intercept)", ]
  
  # Create plot
  p <- ggplot(plot_data, aes(x = reorder(Parameter, Estimate), y = Estimate, 
                             color = Significant)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    coord_flip() +
    labs(title = title, x = "Parameter", y = "Effect Size") +
    theme_minimal() +
    scale_color_manual(values = c("No" = "gray", "Yes" = "blue"))
  
  return(p)
}

# 11. Create and save plots
cat("\n=== CREATING VISUALIZATIONS ===\n")

# Effect plots for each model
p1 <- create_effect_plot(model1, "Model 1: Simple Association Drivers")
p2 <- create_effect_plot(model2, "Model 2: Environmental Stressors")
p3 <- create_effect_plot(model3, "Model 3: Temporal Autocorrelation")

# Combine plots
combined_plot <- grid.arrange(p1, p2, p3, ncol = 2)

# Save plots
ggsave("week6/mcmcglmm_effect_plots.png", combined_plot, width = 12, height = 10)

# 12. Save model results
cat("\n=== SAVING RESULTS ===\n")

# Save model objects
saveRDS(model1, "week6/model1_simple_association.rds")
saveRDS(model2, "week6/model2_environmental.rds")
saveRDS(model3, "week6/model3_temporal.rds")
saveRDS(model4, "week6/model4_state_transitions.rds")
saveRDS(model5, "week6/model5_pair_bond_transitions.rds")

# Save summaries as text files
sink("week6/model_summaries.txt")
cat("=== MCMCglmm MODEL SUMMARIES ===\n\n")
cat("Model 1: Simple Association Drivers\n")
cat("===================================\n")
print(summary_model1)
cat("\n\nModel 2: Environmental Stressors\n")
cat("=================================\n")
print(summary_model2)
cat("\n\nModel 3: Temporal Autocorrelation\n")
cat("==================================\n")
print(summary_model3)
cat("\n\nModel 4: State Transitions\n")
cat("==========================\n")
print(summary_model4)
cat("\n\nModel 5: Pair Bond Transitions\n")
cat("==============================\n")
print(summary_model5)
cat("\n\nModel Comparison (DIC)\n")
cat("======================\n")
print(dics)
sink()


# 13. Create summary report
cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to week6/ folder:\n")
cat("- Model objects (.rds files)\n")
cat("- Model summaries (model_summaries.txt)\n")
cat("- Effect plots (mcmcglmm_effect_plots.png)\n")
cat("- Model comparison table\n")

# Print key findings
cat("\n=== KEY FINDINGS ===\n")
cat("Best model (lowest DIC):", dics$Model[which.min(dics$DIC)], "\n")
cat("DIC:", min(dics$DIC), "\n") 