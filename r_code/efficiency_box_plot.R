library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggstar)
library(patchwork)
ANALYSIS_DIR <- "the-mind/analysis_results"

json_path <- file.path(ANALYSIS_DIR, "overall_performance_llm_time_based_second.json")
raw <- fromJSON(json_path, simplifyVector = FALSE)

flatten_results <- function(raw) {
  rows <- list()
  for (llm in names(raw)) {
    for (player_type in names(raw[[llm]])) {
      inner <- raw[[llm]][[player_type]]
      hand_list <- if ("time_based_second" %in% names(inner)) {
        inner[["time_based_second"]]
      } else {
        inner
      }
      for (hand in names(hand_list)) {
        entry <- hand_list[[hand]]
        entry_success <- if (is.null(entry$success)) NA_real_ else as.numeric(entry$success)
        entry_time <- if (is.null(entry$time)) NA_real_ else as.numeric(entry$time)
        rows[[length(rows) + 1]] <- data.frame(
          llm = llm,
          player_type = player_type,
          hand = as.integer(hand),
          success = entry_success,
          time = entry_time,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  bind_rows(rows)
}

df_llm <- flatten_results(raw)


flatten_flat <- function(flat_raw) {
  rows <- list()
  for (player_type in names(flat_raw)) {
    hand_list <- flat_raw[[player_type]]
    for (hand in names(hand_list)) {
      records <- hand_list[[hand]]
      for (rec in records) {
        rec_success <- if (is.null(rec$success)) NA_real_ else as.numeric(rec$success)
        rec_time <- if (is.null(rec$hesitation_time)) NA_real_ else as.numeric(rec$hesitation_time)
        rows[[length(rows) + 1]] <- data.frame(
          player_type = player_type,
          hand = as.integer(hand),
          success = rec_success,
          time = rec_time,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  bind_rows(rows)
}


human_raw <- fromJSON(file.path(ANALYSIS_DIR, "human_success_time.json"), simplifyVector = FALSE)
df_human <- flatten_flat(human_raw) %>%
  group_by(player_type, hand) %>%
  summarise(success = mean(success, na.rm = TRUE), time = mean(time, na.rm = TRUE), .groups = "drop") %>%
  mutate(llm = "human")


computer_use_raw <- fromJSON(file.path(ANALYSIS_DIR, "gemini_computer_use_success_time.json"), simplifyVector = FALSE)
df_computer_use <- flatten_flat(computer_use_raw) %>%
  group_by(player_type, hand) %>%
  summarise(success = mean(success, na.rm = TRUE), time = mean(time, na.rm = TRUE), .groups = "drop") %>%
  mutate(llm = "computer_use")


gpt6_computer_use_raw <- fromJSON(file.path(ANALYSIS_DIR, "gpt6_computer_use_success_time.json"), simplifyVector = FALSE)
df_gpt6_computer_use <- flatten_flat(gpt6_computer_use_raw) %>%
  group_by(player_type, hand) %>%
  summarise(success = mean(success, na.rm = TRUE), time = mean(time, na.rm = TRUE), .groups = "drop") %>%
  mutate(llm = "gpt6_computer_use")


gptluna_computer_use_raw <- fromJSON(file.path(ANALYSIS_DIR, "gptluna_computer_use_success_time.json"), simplifyVector = FALSE)
df_gptluna_computer_use <- flatten_flat(gptluna_computer_use_raw) %>%
  group_by(player_type, hand) %>%
  summarise(success = mean(success, na.rm = TRUE), time = mean(time, na.rm = TRUE), .groups = "drop") %>%
  mutate(llm = "gptluna_computer_use")

df <- bind_rows(df_llm, df_human, df_computer_use, df_gpt6_computer_use, df_gptluna_computer_use)
df <- df %>% mutate(success = success / 110 * 100)

condition_colors <- c(
  "Random" = "#4B4C4E",
  "Linear\n(-Calibration)" = "#A6CEE3",
  "Linear\n(+Calibration)" = "#1F78B4",
  "Bayesian\n(-Calibration)" = "#FDBF6F",
  "Bayesian\n(+Calibration)" = "#FF7F00",
  "LLM (same)" = "#33A02C",
  "LLM (different)" = "#B2DF8A"
)

player_type_labels <- c(
  "random" = "Random",
  "count_false" = "Linear\n(-Calibration)",
  "count_true" = "Linear\n(+Calibration)",
  "bayesian_false" = "Bayesian\n(-Calibration)",
  "bayesian_true" = "Bayesian\n(+Calibration)",
  "llm_same" = "LLM (same)",
  "llm_different" = "LLM (different)"
)


df <- df %>%
  mutate(player_type = recode(player_type, !!!player_type_labels))


df <- df %>%
  mutate(llm = recode(as.character(llm),
                      "human" = "Human",
                      "computer_use" = "Gemini (CUA)",
                      "gpt6_computer_use" = "GPT-6 (CUA)",
                      "gptluna_computer_use" = "GPT-5.6-Luna (CUA)"))


df <- df %>% filter(!player_type %in% c("LLM (same)", "LLM (different)"))


df <- df %>%
  mutate(
    success_z = as.numeric(scale(success)),
    
    time_z = as.numeric(scale(time)),
    composite = 0.5 * success_z - 0.5 * time_z
  )


combo_summary <- df %>%
  group_by(llm, player_type) %>%
  summarise(
    mean_composite = mean(composite, na.rm = TRUE),
    n_hands = sum(!is.na(composite)),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_composite))

print(combo_summary)

top10_combos <- combo_summary %>% slice_head(n = 20)


combo_order <- paste(top10_combos$llm, top10_combos$player_type, sep = " × ")

df_top10 <- df %>%
  inner_join(top10_combos %>% select(llm, player_type), by = c("llm", "player_type")) %>%
  mutate(combo = factor(paste(llm, player_type, sep = " × "), levels = combo_order))

box_plot <- ggplot(
  df_top10,
  aes(x = combo, y = composite, fill = player_type, color = player_type)
) +
  geom_jitter(
    width = 0.15,
    size = 0.8,
    alpha = 0.25,
    color = "grey40"
  ) +
  geom_boxplot(
    alpha = 0.7,
    outlier.shape = NA,
    width = 0.6
  ) +
  scale_fill_manual(values = condition_colors) +
  scale_color_manual(
    values = condition_colors,
    guide = "none"
  ) +
  scale_x_discrete(drop = FALSE) +
  labs(
    x = NULL,
    y = "Efficiency Score\n(higher = better)",
    fill = "Partner type"
  ) +
  theme_bw(base_size = 16)

shape_values_10 <- c(
  gemini  = 13,
  gpt6    = 6,
  gpt56   = 5,
  human   = 11,
  qwen    = 13,
  llama70 = 23,
  gpt120  = 15
)

shape_axis_df <- top10_combos %>%
  mutate(
    combo = factor(combo_order, levels = combo_order),
    
    model = case_when(
      grepl("gemini-3\\.7-flash", llm, ignore.case = TRUE) ~ "gemini",
      grepl("gpt-6-astra", llm, ignore.case = TRUE) ~ "gpt6",
      grepl("gpt-5\\.6-luna", llm, ignore.case = TRUE) ~ "gpt56",
      grepl("qwen3-32b", llm, ignore.case = TRUE) ~ "qwen",
      grepl("llama.*70b", llm, ignore.case = TRUE) ~ "llama70",
      grepl("gpt-oss.*120b", llm, ignore.case = TRUE) ~ "gpt120",
      grepl("human", llm, ignore.case = TRUE) ~ "human"
    ),
    
    model = factor(
      model,
      levels = names(shape_values_10)
    ),
    
    fill_color = ifelse(
      model %in% c("qwen", "llama70", "gpt120"),
      NA,
      "black"
    )
  )

shape_axis <- ggplot(
  shape_axis_df,
  aes(
    x = combo,
    y = 1,
    starshape = model,
    fill = fill_color
  )
) +
  geom_star(
    size = 5,
    color = "black"
  ) +
  scale_starshape_manual(
    values = shape_values_10,
    guide = "none"
  ) +
  scale_fill_identity() +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(limits = c(0.5, 1.5)) +
  theme_void()

box_plot_no_labels <- box_plot +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.x = element_blank()
  )

final_plot <- box_plot_no_labels / shape_axis +
  patchwork::plot_layout(
    heights = c(1, 0.15)
  )

print(final_plot)

ggsave(
  file.path(ANALYSIS_DIR, "top10_efficiency.pdf"),
  final_plot,
  width = 15.38,
  height = 2.3,
  dpi = 300
)


pretty_model <- function(llm) {
  case_when(
    grepl("\\(CUA\\)", llm)                                ~ llm,
    llm == "Human"                                          ~ "Human",
    grepl("gemini-3\\.7-flash", llm, ignore.case = TRUE)    ~ "Gemini-3.7-Flash",
    grepl("gpt-6-astra",        llm, ignore.case = TRUE)    ~ "GPT-6-Astra",
    grepl("gpt-5\\.6-luna",     llm, ignore.case = TRUE)    ~ "GPT-5.6-Luna",
    grepl("qwen3-32b",          llm, ignore.case = TRUE)    ~ "Qwen3-32B",
    grepl("llama.*70b",         llm, ignore.case = TRUE)    ~ "Llama-70B",
    grepl("gpt-oss.*120b",      llm, ignore.case = TRUE)    ~ "GPT-OSS-120B",
    TRUE                                                    ~ llm
  )
}


combo_labels <- setNames(pretty_model(top10_combos$llm), combo_order)

final_plot <- ggplot(
  df_top10,
  aes(x = combo, y = composite, fill = player_type, color = player_type)
) +
  geom_jitter(width = 0.15, size = 0.8, alpha = 0.25, color = "grey40") +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.6) +
  scale_fill_manual(values = condition_colors) +
  scale_color_manual(values = condition_colors, guide = "none") +
  scale_x_discrete(drop = FALSE, labels = combo_labels) +
  labs(
    x = NULL,
    y = "Efficiency Score\n(higher = better)",
    fill = "Partner type"
  ) +
  theme_bw(base_size = 16) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

print(final_plot)

ggsave(
  file.path(ANALYSIS_DIR, "top20_efficiency.pdf"),
  final_plot,
  width = 12,
  height = 4.25,
  dpi = 300
)