# supercolony_dyad_composition_analysis.R
# Analyze and visualize sex-pair composition and network metrics in supercolony vs. non-supercolony dyads

library(tidyverse)

# Load the integrated dyad dataset
df <- read_csv("dyad_full_integrated.csv")

# Ensure in_supercolony is logical and not NA
df <- df %>%
  mutate(in_supercolony = ifelse(is.na(in_supercolony), FALSE, in_supercolony))

# Summarize sex-pair composition and network metrics
composition_summary <- df %>%
  group_by(in_supercolony, sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_degree = mean(degree_bird1, na.rm = TRUE),
    mean_betweenness = mean(betweenness_bird1, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    mean_percent_time = mean(percent_time.x, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  group_by(in_supercolony) %>%
  mutate(prop_dyads = n_dyads / sum(n_dyads))

print(composition_summary)

# Visualize sex-pair proportions in supercolony vs. non-supercolony
ggplot(composition_summary, aes(x = sex_pair, y = prop_dyads, fill = in_supercolony)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Sex-pair composition: Supercolony vs. Non-supercolony", y = "Proportion of Dyads", x = "Sex Pair") +
  scale_fill_manual(values = c("#999999", "#E69F00"), labels = c("Non-supercolony", "Supercolony")) +
  theme_minimal() 

# ----
# Quantitative analysis for the most robust 2016 supercolony (SC_2016_2, SPRA_42, 2016)
sc2016 <- df %>%
  filter(supercolony_id == "SC_2016_2", colony == "SPRA_42", year == 2016)

sc2016_summary <- sc2016 %>%
  group_by(sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_degree = mean(degree_bird1, na.rm = TRUE),
    mean_betweenness = mean(betweenness_bird1, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    mean_percent_time = mean(percent_time.x, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(prop_dyads = n_dyads / sum(n_dyads))

cat("\nSummary for SC_2016_2 (SPRA_42, 2016):\n")
print(sc2016_summary)

# ----
# Summary analysis across all supercolonies (2015–2017)
all_sc <- df %>%
  filter(year %in% 2015:2017, !is.na(supercolony_id))

all_sc_summary <- all_sc %>%
  group_by(supercolony_id, colony, year, sex_pair) %>%
  summarise(
    n_dyads = n(),
    mean_degree = mean(degree_bird1, na.rm = TRUE),
    mean_betweenness = mean(betweenness_bird1, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    mean_percent_time = mean(percent_time.x, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nSummary by supercolony, colony, year, and sex_pair:\n")
print(all_sc_summary, n = 50)

supercolony_level <- all_sc %>%
  group_by(supercolony_id, colony, year) %>%
  summarise(
    n_birds = n_distinct(c(bird1, bird2)),
    n_dyads = n(),
    mean_degree = mean(degree_bird1, na.rm = TRUE),
    mean_betweenness = mean(betweenness_bird1, na.rm = TRUE),
    mean_dyad_strength = mean(dyad_strength, na.rm = TRUE),
    mean_percent_time = mean(percent_time.x, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nSupercolony-level summary (all sex-pairs):\n")
print(supercolony_level, n = 50) 

# ----
# Test: For each supercolony, is ff the sex_pair with the highest mean betweenness?

# For each supercolony/colony/year, find the sex_pair with the highest mean betweenness
ff_highest_betweenness <- all_sc %>%
  group_by(supercolony_id, colony, year, sex_pair) %>%
  summarise(
    mean_betweenness = mean(betweenness_bird1, na.rm = TRUE),
    n_dyads = n(),
    .groups = 'drop'
  ) %>%
  group_by(supercolony_id, colony, year) %>%
  arrange(desc(mean_betweenness)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(ff_is_highest = sex_pair == "ff")

cat("\nSupercolonies where FF has the highest mean betweenness:\n")
print(ff_highest_betweenness)

n_ff_highest <- sum(ff_highest_betweenness$ff_is_highest, na.rm = TRUE)
cat("\nNumber of supercolony/colony/year groups where FF is highest in mean betweenness:", n_ff_highest, "\n") 

# ----
# Compare anchor sex_pair in regular colonies vs. supercolonies

# 1. For regular colonies (not in supercolony)
colony_dyads <- df %>%
  filter(is.na(supercolony_id)) %>%
  group_by(colony, year, sex_pair) %>%
  summarise(
    mean_betweenness = mean(betweenness_bird1, na.rm = TRUE),
    n_dyads = n(),
    .groups = 'drop'
  ) %>%
  group_by(colony, year) %>%
  arrange(desc(mean_betweenness)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(context = "colony")

# 2. For supercolonies (already calculated as ff_highest_betweenness)
supercolony_anchors <- ff_highest_betweenness %>%
  mutate(context = "supercolony")

# 3. Combine and tabulate
anchors_combined <- bind_rows(colony_dyads, supercolony_anchors)

# 4. Tabulate frequency of each sex_pair as anchor by context
anchor_freq <- anchors_combined %>%
  group_by(context, sex_pair) %>%
  summarise(n = n(), .groups = 'drop') %>%
  group_by(context) %>%
  mutate(prop = n / sum(n))

cat("\nFrequency of each sex_pair as anchor (highest mean betweenness) by context:\n")
print(anchor_freq)

# 5. Chi-squared test
library(tidyr)
contingency <- anchors_combined %>%
  count(context, sex_pair) %>%
  pivot_wider(names_from = sex_pair, values_from = n, values_fill = 0)

cat("\nContingency table (context x sex_pair):\n")
print(contingency)

chisq_test <- chisq.test(as.matrix(contingency[,-1]))
cat("\nChi-squared test for difference in anchor sex_pair distribution between colonies and supercolonies:\n")
print(chisq_test) 