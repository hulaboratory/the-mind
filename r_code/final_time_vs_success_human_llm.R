library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

library(ggstar)
library(cowplot)

results_dir <- "the-mind/analysis_results"
json_path <- "the-mind/analysis_results/overall_performance_llm_time_based_second.json"
raw <- fromJSON(json_path, simplifyVector = FALSE)

check_missing <- function(raw) {
  problems <- list()
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
        missing_fields <- c()
        if (is.null(entry) || length(entry) == 0) {
          missing_fields <- c(missing_fields, "entry_empty")
        } else {
          if (is.null(entry$success)) missing_fields <- c(missing_fields, "success")
          if (is.null(entry$time)) missing_fields <- c(missing_fields, "time")
        }
        if (length(missing_fields) > 0) {
          problems[[length(problems) + 1]] <- data.frame(
            llm = llm, player_type = player_type, hand = hand,
            missing = paste(missing_fields, collapse = ", "),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  bind_rows(problems)
}

missing_report <- check_missing(raw)
print(missing_report)

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


check_missing_flat <- function(flat_raw) {
  problems <- list()
  for (player_type in names(flat_raw)) {
    hand_list <- flat_raw[[player_type]]
    for (hand in names(hand_list)) {
      records <- hand_list[[hand]]
      for (i in seq_along(records)) {
        rec <- records[[i]]
        missing_fields <- c()
        if (is.null(rec) || length(rec) == 0) {
          missing_fields <- c(missing_fields, "record_empty")
        } else {
          if (is.null(rec$success)) missing_fields <- c(missing_fields, "success")
          if (is.null(rec$hesitation_time)) missing_fields <- c(missing_fields, "hesitation_time")
        }
        if (length(missing_fields) > 0) {
          problems[[length(problems) + 1]] <- data.frame(
            player_type = player_type, hand = hand, record_index = i,
            missing = paste(missing_fields, collapse = ", "),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  bind_rows(problems)
}

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

load_flat <- function(path, llm_label) {
  flat_raw <- fromJSON(path, simplifyVector = FALSE)
  print(check_missing_flat(flat_raw))
  flatten_flat(flat_raw) %>%
    group_by(player_type, hand) %>%
    summarise(
      success = mean(success, na.rm = TRUE),
      time = mean(time, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(llm = llm_label)
}



df_human <- load_flat(file.path(results_dir, "human_success_time.json"), "human")

df_computer_use         <- load_flat(file.path(results_dir, "gemini_computer_use_success_time.json"), "computer_use")
df_GPT6_computer_use    <- load_flat(file.path(results_dir, "gpt6_computer_use_success_time.json"), "GPT6_computer_use")
df_GPTluna_computer_use <- load_flat(file.path(results_dir, "gptluna_computer_use_success_time.json"), "GPTluna_computer_use")



df <- bind_rows(
  df_llm                  %>% mutate(source = "LLM"),
  df_human                %>% mutate(source = "Human"),
  df_computer_use         %>% mutate(source = "CUA"),
  df_GPT6_computer_use    %>% mutate(source = "CUA"),
  df_GPTluna_computer_use %>% mutate(source = "CUA")
)

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


# Shape rules only apply to the regular LLM models (source == "LLM").
llm_levels_no_human <- sort(unique(df$llm[df$source == "LLM"]))


model_shape_patterns <- list(
  list(pattern = "gemini",             starshape = 12),
  list(pattern = "gpt-5\\.6",          starshape = 1), 
  list(pattern = "gpt-6.*astra",       starshape = 14), 
  list(pattern = "gpt-oss.*120b",      starshape = 11),
  list(pattern = "gpt-oss.*[^0-9]20b", starshape = 23),
  list(pattern = "llama.*70b",         starshape = 28),
  list(pattern = "llama.*[^0-9]8b",    starshape = 5),
  list(pattern = "llama.*[^0-9]1b",    starshape = 6),
  list(pattern = "qwen.*32b",          starshape = 4),
  list(pattern = "qwen.*[^0-9]8b",     starshape = 13)
)

shape_values <- setNames(rep(NA_real_, length(llm_levels_no_human)), llm_levels_no_human)
for (p in model_shape_patterns) {
  idx <- grepl(p$pattern, names(shape_values), ignore.case = TRUE)
  shape_values[idx] <- p$starshape
}
if (any(is.na(shape_values))) {
  warning("The following models did not match any shape rule: ",
          paste(names(shape_values)[is.na(shape_values)], collapse = ", "))
}


clean_model_name <- function(x) {
  x <- gsub("gpt", "GPT", x, ignore.case = TRUE)
  x <- gsub("[-_]instruct", "", x, ignore.case = TRUE)   # llama-70b-instruct -> llama-70b
  x
}

names(shape_values) <- clean_model_name(names(shape_values))
df <- df %>% mutate(llm = clean_model_name(llm))

if (anyDuplicated(names(shape_values))) {
  stop("Name collision after cleaning: ",
       paste(names(shape_values)[duplicated(names(shape_values))], collapse = ", "))
}



find_one <- function(pat) {
  hit <- grep(pat, names(shape_values), ignore.case = TRUE, value = TRUE)
  if (length(hit) != 1) stop("Pattern '", pat, "' matched ", length(hit), " models: ",
                             paste(hit, collapse = ", "))
  hit
}


cua_map <- c(
  "computer_use"         = find_one("gemini"),
  "GPT6_computer_use"    = find_one("gpt-6.*astra"),
  "GPTluna_computer_use" = find_one("gpt-5\\.6")
)
print(cua_map)

df <- df %>%
  mutate(llm = as.character(llm),
         llm = ifelse(llm == "human", "Human", llm),
         llm = ifelse(llm %in% names(cua_map), unname(cua_map[llm]), llm))

shape_values_all <- c("Human" = 15, shape_values)
if (anyDuplicated(shape_values_all)) {
  warning("Duplicate starshapes: ",
          paste(names(shape_values_all)[duplicated(shape_values_all) |
                                          duplicated(shape_values_all, fromLast = TRUE)],
                collapse = ", "))
}

df <- df %>% mutate(llm = factor(llm, levels = names(shape_values_all)))
stopifnot(!any(is.na(df$llm)))

solid_pat <- "gemini|gpt-5\\.6|gpt-6.*astra"
df <- df %>%
  mutate(fill_color = ifelse(
    llm == "Human" | grepl(solid_pat, llm, ignore.case = TRUE),
    as.character(player_type), NA_character_
  ))


summary_df <- df %>%
  group_by(source, llm, player_type) %>%
  summarise(
    n = n(),
    success_mean = mean(success, na.rm = TRUE),
    success_se = qt(0.975, df = n - 1) * sd(success, na.rm = TRUE) / sqrt(n),
    time_mean = mean(time, na.rm = TRUE),
    time_se = qt(0.975, df = n - 1) * sd(time, na.rm = TRUE) / sqrt(n),
    fill_color = fill_color[1],
    .groups = "drop"
  )


ref <- summary_df %>%
  filter(source == "Human", player_type == "Bayesian\n(+Calibration)")
stopifnot(nrow(ref) == 1)
ref_x <- ref$time_mean
ref_y <- ref$success_mean




df_human_plot        <- df %>% filter(source == "Human")
df_computer_use_plot <- df %>% filter(source == "CUA")
df_llm_only          <- df %>% filter(source == "LLM")

summary_human_plot        <- summary_df %>% filter(source == "Human")
summary_computer_use_plot <- summary_df %>% filter(source == "CUA")
summary_llm_only          <- summary_df %>% filter(source == "LLM")


# ranges computed over ALL panels' data so axes line up
x_range <- range(
  c(df$time,
    summary_df$time_mean - summary_df$time_se,
    summary_df$time_mean + summary_df$time_se),
  na.rm = TRUE
)
y_range <- range(
  c(df$success,
    summary_df$success_mean - summary_df$success_se,
    summary_df$success_mean + summary_df$success_se),
  na.rm = TRUE
)

make_panel <- function(pts, smry, title, y_lab = NULL, star_size=2, mean_size = 5, hide_y_text = TRUE,
                       solid_stroke = 0.1, show_ref_label = FALSE) {
  pts  <- pts  %>% mutate(panel = title)
  smry <- smry %>% mutate(panel = title)
  
  smry_hollow <- smry %>% filter(is.na(fill_color))
  smry_solid  <- smry %>% filter(!is.na(fill_color))
  
  p <- ggplot() +

    annotate("rect", xmin = -Inf, xmax = ref_x, ymin = ref_y, ymax = Inf,
             fill = "#FFF3B0", alpha = 0.45) +
    geom_vline(xintercept = ref_x, linetype = "dashed", color = "grey40", linewidth = 0.5) +
    geom_hline(yintercept = ref_y, linetype = "dashed", color = "grey40", linewidth = 0.5)
  
  if (show_ref_label) {
    dx <- diff(x_range); dy <- diff(y_range)
    lab_x <- ref_x + 0.10 * dx
    lab_y <- y_range[2] - 0.10 * dy 
    p <- p +
      annotate("segment",
               x = lab_x - 0.01 * dx, xend = ref_x - 0.06 * dx,
               y = lab_y, yend = lab_y,
               arrow = arrow(length = unit(0.18, "cm"), type = "closed"),
               color = "grey25", linewidth = 0.5) +
      annotate("text", x = lab_x, y = lab_y,
               label = "More efficient\nthan humans",
               hjust = 0, vjust = 0.5, size = 4.5, color = "grey25",
               fontface = "italic", lineheight = 0.9)
  }
  
  p <- p +
    geom_star(
      data = pts,
      aes(x = time, y = success, color = player_type, starshape = llm, fill = fill_color),
      alpha = 0.15, size = star_size, show.legend = FALSE
    ) +
    geom_segment(
      data = smry,
      aes(x = time_mean - time_se, xend = time_mean + time_se,
          y = success_mean, yend = success_mean, color = player_type),
      linewidth = 0.7, show.legend = FALSE
    ) +
    geom_segment(
      data = smry,
      aes(x = time_mean, xend = time_mean,
          y = success_mean - success_se, yend = success_mean + success_se,
          color = player_type),
      linewidth = 0.7, show.legend = FALSE
    ) +
    geom_star(
      data = smry_hollow,
      aes(x = time_mean, y = success_mean, color = player_type, starshape = llm),
      fill = NA, size = mean_size, show.legend = FALSE
    )
  

  if (solid_stroke > 0) {
    p <- p + geom_star(
      data = smry_solid,
      aes(x = time_mean, y = success_mean, starshape = llm, fill = fill_color),
      color = "white", starstroke = solid_stroke, size = mean_size, show.legend = FALSE
    )
  } else {
    p <- p + geom_star(
      data = smry_solid,
      aes(x = time_mean, y = success_mean, color = player_type, starshape = llm, fill = fill_color),
      size = mean_size, show.legend = FALSE
    )
  }
  
  p <- p +
    labs(title=title) +
    labs(x = "Total play time (seconds)\n(faster is better ←)", y = y_lab) +
    scale_color_manual(values = condition_colors, guide = "none") +
    scale_fill_manual(values = condition_colors, na.value = NA, guide = "none") +
    scale_starshape_manual(values = shape_values_all, drop = FALSE, guide = "none") +
    scale_x_continuous(limits = x_range) +
    scale_y_continuous(limits = y_range) +
    theme_bw(base_size = 18) +
    theme(legend.position = "none",
          strip.text = element_text(size = 16),
          plot.title = element_text(size = 18))
  if (hide_y_text) p <- p + theme(axis.text.y = element_blank())
  p
}


p_human <- make_panel(
  df_human_plot, summary_human_plot, title = "Human",
  y_lab = "% successfully played cards\n(higher is better →)",
  mean_size = 5, hide_y_text = FALSE, show_ref_label=TRUE
)
p_human


p_llm <- make_panel(df_llm_only, summary_llm_only, star_size=1, mean_size=4,title = "LLM")
p_llm


p_computer_use <- make_panel(df_computer_use_plot, summary_computer_use_plot, title = "CUA")
p_computer_use



llm_is_solid <- names(shape_values_all) == "Human" |
  grepl(solid_pat, names(shape_values_all), ignore.case = TRUE)
llm_fill_override <- ifelse(llm_is_solid, "black", NA)


shape_legend_plot <- ggplot(df, aes(x = time, y = success, starshape = llm, fill = fill_color)) +
  geom_star(size = 2) +
  scale_fill_manual(values = condition_colors, na.value = NA, guide = "none") +
  scale_starshape_manual(values = shape_values_all, drop = FALSE, name = "Player type") +
  guides(starshape = guide_legend(override.aes = list(
    fill = llm_fill_override,
    colour = ifelse(llm_is_solid, "white", "black"),
    starstroke = ifelse(llm_is_solid, 0.4, 0.5),
    alpha = 1, size = 4
  ))) +
  theme_bw(base_size = 18) +
  theme(
    legend.position = "right",
    legend.justification = "top",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 15)
  )

partner_legend_plot <- ggplot(df, aes(x = time, y = success, color = player_type)) +
  geom_point(alpha = 0, size = 0) +
  scale_color_manual(values = condition_colors, name = "Partner type") +
  guides(color = guide_legend(override.aes = list(alpha = 1, size = 3))) +
  theme_bw(base_size = 18) +
  theme(
    legend.position = "right",
    legend.justification = "top",
    legend.text = element_text(size = 14, margin = margin(t = 4, b = 4)),
    legend.title = element_text(size = 15)
    # legend.key.height = unit(0.9, "cm")
  )


# combined
shape_legend   <- suppressWarnings(cowplot::get_legend(shape_legend_plot))
partner_legend <- suppressWarnings(cowplot::get_legend(partner_legend_plot))


w_shape   <- grid::convertWidth(sum(shape_legend$widths),   "cm", valueOnly = TRUE)
w_partner <- grid::convertWidth(sum(partner_legend$widths), "cm", valueOnly = TRUE)
gap <- -0.1   

legends_two_col <- gtable::gtable_row(
  "legends",
  list(shape_legend, grid::nullGrob(), partner_legend),
  height = unit(1, "null"),
  widths = unit(c(w_shape, gap, w_partner), "cm")
)

combined <- p_human + p_computer_use + p_llm + wrap_elements(full = legends_two_col) +
  plot_layout(
    axis_titles = "collect",
    widths = unit(c(1, 1, 1, w_shape + gap + w_partner),
                  c("null", "null", "null", "cm"))
  )

combined

ggsave("human_llm_use_combined.pdf", combined, width = 14.64, height = 4.27, dpi = 300,
       device = cairo_pdf)