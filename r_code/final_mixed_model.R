library(jsonlite)
library(dplyr)
library(tidyr)
library(purrr)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(ggplot2)
library(tibble)
library(ggstar)

ANALYSIS_DIR <- "the-mind/analysis_results"
llm_data_raw <- fromJSON(file.path(ANALYSIS_DIR, "success_by_decile.json"), simplifyVector = FALSE)

llm_decile <- imap_dfr(llm_data_raw, function(partner_data, model_name) {
  imap_dfr(partner_data, function(prompt_data, partner_type) {
    imap_dfr(prompt_data, function(hand_data, prompt_type) {
      imap_dfr(hand_data, function(cell, hand_id) {
        tibble(
          model = model_name,
          partner_type = partner_type,
          prompt_type = prompt_type,
          hand_id = hand_id,
          participant = NA_character_,
          decile = 1:10,
          success = as.numeric(unlist(cell$success_by_decile)),
          total = as.numeric(unlist(cell$total_by_decile))
        )
      })
    })
  })
})

human_data_raw <- fromJSON(file.path(ANALYSIS_DIR, "human_success_by_decile.json"), simplifyVector = FALSE)

human_decile <- imap_dfr(human_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    tibble(
      model = "Human",
      partner_type = partner_type,
      prompt_type = "human",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      decile = 1:10,
      success = as.numeric(unlist(cell$success_by_decile)),
      total = as.numeric(unlist(cell$total_by_decile))
    )
  })
})


computer_use_data_raw <- fromJSON(file.path(ANALYSIS_DIR, "gemini_success_by_decile.json"), simplifyVector = FALSE)

computer_use_decile <- imap_dfr(computer_use_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    tibble(
      model = "computer_use",
      partner_type = partner_type,
      prompt_type = "computer_use",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      decile = 1:10,
      success = as.numeric(unlist(cell$success_by_decile)),
      total = as.numeric(unlist(cell$total_by_decile))
    )
  })
})

gpt6_computer_use_data_raw <- fromJSON(file.path(ANALYSIS_DIR, "gpt6_success_by_decile.json"), simplifyVector = FALSE)

gpt6_computer_use_decile <- imap_dfr(gpt6_computer_use_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    tibble(
      model = "gpt6_computer_use",
      partner_type = partner_type,
      prompt_type = "gpt6_computer_use",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      decile = 1:10,
      success = as.numeric(unlist(cell$success_by_decile)),
      total = as.numeric(unlist(cell$total_by_decile))
    )
  })
})

gptluna_computer_use_data_raw <- fromJSON(file.path(ANALYSIS_DIR, "gptluna_success_by_decile.json"), simplifyVector = FALSE)

gptluna_computer_use_decile <- imap_dfr(gptluna_computer_use_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    tibble(
      model = "gptluna_computer_use",
      partner_type = partner_type,
      prompt_type = "gptluna_computer_use",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      decile = 1:10,
      success = as.numeric(unlist(cell$success_by_decile)),
      total = as.numeric(unlist(cell$total_by_decile))
    )
  })
})

decile_data <- bind_rows(llm_decile, human_decile, computer_use_decile,
                         gpt6_computer_use_decile, gptluna_computer_use_decile) %>%
  mutate(hand_id = factor(hand_id), participant = factor(participant),
         prompt_type_re = factor(prompt_type))

fit_success_llm_pooled <- function(d) {
  d <- d %>% filter(total > 0)
  n_prompt <- n_distinct(d$prompt_type_re)
  if (nrow(d) == 0 || n_distinct(d$hand_id) < 2 || n_distinct(d$decile) < 2) {
    return(tibble(success_slope = NA_real_, success_p = NA_real_))
  }
  tryCatch({
    formula_str <- if (n_prompt >= 2) {
      "cbind(success, total - success) ~ decile + (1 | prompt_type_re) + (1 | hand_id)"
    } else {
      "cbind(success, total - success) ~ decile + (1 | hand_id)"
    }
    model <- suppressWarnings(glmer(as.formula(formula_str), data = d, family = binomial))
    broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "decile") %>%
      transmute(success_slope = estimate, success_p = p.value)
  }, error = function(e) tibble(success_slope = NA_real_, success_p = NA_real_))
}

fit_success_human <- function(d) {
  d <- d %>% filter(total > 0)
  if (nrow(d) == 0 || n_distinct(d$participant) < 2 || n_distinct(d$decile) < 2) {
    return(tibble(success_slope = NA_real_, success_p = NA_real_))
  }
  tryCatch({
    model <- suppressWarnings(
      glmer(cbind(success, total - success) ~ decile + (1 | participant) + (1 | hand_id),
            data = d, family = binomial)
    )
    broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "decile") %>%
      transmute(success_slope = estimate, success_p = p.value)
  }, error = function(e) tibble(success_slope = NA_real_, success_p = NA_real_))
}


fit_success_computer_use <- function(d) {
  d <- d %>% filter(total > 0)
  if (nrow(d) == 0 || n_distinct(d$hand_id) < 2 || n_distinct(d$decile) < 2) {
    return(tibble(success_slope = NA_real_, success_p = NA_real_))
  }
  tryCatch({
    model <- suppressWarnings(
      glmer(cbind(success, total - success) ~ decile + (1 | hand_id),
            data = d, family = binomial)
    )
    broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "decile") %>%
      transmute(success_slope = estimate, success_p = p.value)
  }, error = function(e) tibble(success_slope = NA_real_, success_p = NA_real_))
}


success_llm_groups <- llm_decile %>% distinct(model, partner_type)
success_llm_results <- pmap_dfr(success_llm_groups, function(model, partner_type) {
  d <- decile_data %>% filter(model == !!model, partner_type == !!partner_type)
  fit <- fit_success_llm_pooled(d)
  tibble(model = model, partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p)
})

success_human_groups <- human_decile %>% distinct(partner_type)
success_human_results <- pmap_dfr(success_human_groups, function(partner_type) {
  d <- human_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_human(d)
  tibble(model = "Human", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p)
})

success_computer_use_groups <- computer_use_decile %>% distinct(partner_type)
success_computer_use_results <- pmap_dfr(success_computer_use_groups, function(partner_type) {
  d <- computer_use_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_computer_use(d)
  tibble(model = "computer_use", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p)
})

success_gpt6_computer_use_groups <- gpt6_computer_use_decile %>% distinct(partner_type)
success_gpt6_computer_use_results <- pmap_dfr(success_gpt6_computer_use_groups, function(partner_type) {
  d <- gpt6_computer_use_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_computer_use(d)
  tibble(model = "gpt6_computer_use", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p)
})

success_gptluna_computer_use_groups <- gptluna_computer_use_decile %>% distinct(partner_type)
success_gptluna_computer_use_results <- pmap_dfr(success_gptluna_computer_use_groups, function(partner_type) {
  d <- gptluna_computer_use_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_computer_use(d)
  tibble(model = "gptluna_computer_use", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p)
})

success_results <- bind_rows(success_llm_results, success_human_results,
                             success_computer_use_results, success_gpt6_computer_use_results,
                             success_gptluna_computer_use_results)



llm_by_play <- imap_dfr(llm_data_raw, function(partner_data, model_name) {
  imap_dfr(partner_data, function(prompt_data, partner_type) {
    imap_dfr(prompt_data, function(hand_data, prompt_type) {
      imap_dfr(hand_data, function(cell, hand_id) {
        alpha_vals <- sapply(cell$alpha, function(v) if (is.null(v)) NA_real_ else as.numeric(v))
        n_plays <- length(alpha_vals)
        if (n_plays == 0) return(tibble())
        tibble(
          model = model_name,
          partner_type = partner_type,
          prompt_type = prompt_type,
          hand_id = hand_id,
          participant = NA_character_,
          play_fraction = seq_len(n_plays) / n_plays,
          alpha = alpha_vals
        )
      })
    })
  })
}) %>%
  mutate(prompt_type_re = factor(prompt_type))

human_by_play <- imap_dfr(human_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    alpha_vals <- sapply(cell$alpha, function(v) if (is.null(v)) NA_real_ else as.numeric(v))
    n_plays <- length(alpha_vals)
    if (n_plays == 0) return(tibble())
    tibble(
      model = "Human",
      partner_type = partner_type,
      prompt_type = "human",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      play_fraction = seq_len(n_plays) / n_plays,
      alpha = alpha_vals
    )
  })
})

computer_use_by_play <- imap_dfr(computer_use_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    alpha_vals <- sapply(cell$alpha, function(v) if (is.null(v)) NA_real_ else as.numeric(v))
    n_plays <- length(alpha_vals)
    if (n_plays == 0) return(tibble())
    tibble(
      model = "computer_use",
      partner_type = partner_type,
      prompt_type = "computer_use",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      play_fraction = seq_len(n_plays) / n_plays,
      alpha = alpha_vals
    )
  })
})

gpt6_computer_use_by_play <- imap_dfr(gpt6_computer_use_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    alpha_vals <- sapply(cell$alpha, function(v) if (is.null(v)) NA_real_ else as.numeric(v))
    n_plays <- length(alpha_vals)
    if (n_plays == 0) return(tibble())
    tibble(
      model = "gpt6_computer_use",
      partner_type = partner_type,
      prompt_type = "gpt6_computer_use",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      play_fraction = seq_len(n_plays) / n_plays,
      alpha = alpha_vals
    )
  })
})

gptluna_computer_use_by_play <- imap_dfr(gptluna_computer_use_data_raw, function(participant_data, partner_type) {
  imap_dfr(participant_data, function(cell, key) {
    alpha_vals <- sapply(cell$alpha, function(v) if (is.null(v)) NA_real_ else as.numeric(v))
    n_plays <- length(alpha_vals)
    if (n_plays == 0) return(tibble())
    tibble(
      model = "gptluna_computer_use",
      partner_type = partner_type,
      prompt_type = "gptluna_computer_use",
      hand_id = sub(".*_", "", key),
      participant = sub("_[^_]*$", "", key),
      play_fraction = seq_len(n_plays) / n_plays,
      alpha = alpha_vals
    )
  })
})

fit_alpha_llm_pooled <- function(d) {
  d <- d %>% filter(!is.na(alpha))
  n_prompt <- n_distinct(d$prompt_type_re)
  if (nrow(d) == 0 || n_distinct(d$hand_id) < 2 || n_distinct(d$play_fraction) < 2) {
    return(tibble(alpha_slope = NA_real_, alpha_p = NA_real_))
  }
  tryCatch({
    formula_str <- if (n_prompt >= 2) {
      "alpha ~ play_fraction + (1 | prompt_type_re) + (1 | hand_id)"
    } else {
      "alpha ~ play_fraction + (1 | hand_id)"
    }
    model <- suppressWarnings(lmer(as.formula(formula_str), data = d, REML = FALSE))
    broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "play_fraction") %>%
      transmute(alpha_slope = estimate, alpha_p = p.value)
  }, error = function(e) tibble(alpha_slope = NA_real_, alpha_p = NA_real_))
}

fit_alpha_human <- function(d) {
  d <- d %>% filter(!is.na(alpha))
  if (nrow(d) == 0 || n_distinct(d$participant) < 2 || n_distinct(d$play_fraction) < 2) {
    return(tibble(alpha_slope = NA_real_, alpha_p = NA_real_))
  }
  tryCatch({
    model <- suppressWarnings(
      lmer(alpha ~ play_fraction + (1 | participant) + (1 | hand_id), data = d, REML = FALSE)
    )
    broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "play_fraction") %>%
      transmute(alpha_slope = estimate, alpha_p = p.value)
  }, error = function(e) tibble(alpha_slope = NA_real_, alpha_p = NA_real_))
}


fit_alpha_computer_use <- function(d) {
  d <- d %>% filter(!is.na(alpha))
  if (nrow(d) == 0 || n_distinct(d$hand_id) < 2 || n_distinct(d$play_fraction) < 2) {
    return(tibble(alpha_slope = NA_real_, alpha_p = NA_real_))
  }
  tryCatch({
    model <- suppressWarnings(
      lmer(alpha ~ play_fraction + (1 | hand_id), data = d, REML = FALSE)
    )
    broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "play_fraction") %>%
      transmute(alpha_slope = estimate, alpha_p = p.value)
  }, error = function(e) tibble(alpha_slope = NA_real_, alpha_p = NA_real_))
}

alpha_llm_groups <- llm_by_play %>% distinct(model, partner_type)
alpha_llm_results <- pmap_dfr(alpha_llm_groups, function(model, partner_type) {
  d <- llm_by_play %>% filter(model == !!model, partner_type == !!partner_type)
  fit <- fit_alpha_llm_pooled(d)
  tibble(model = model, partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p)
})

alpha_human_groups <- human_by_play %>% distinct(partner_type)
alpha_human_results <- pmap_dfr(alpha_human_groups, function(partner_type) {
  d <- human_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_human(d)
  tibble(model = "Human", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p)
})

alpha_computer_use_groups <- computer_use_by_play %>% distinct(partner_type)
alpha_computer_use_results <- pmap_dfr(alpha_computer_use_groups, function(partner_type) {
  d <- computer_use_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_computer_use(d)
  tibble(model = "computer_use", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p)
})

alpha_gpt6_computer_use_groups <- gpt6_computer_use_by_play %>% distinct(partner_type)
alpha_gpt6_computer_use_results <- pmap_dfr(alpha_gpt6_computer_use_groups, function(partner_type) {
  d <- gpt6_computer_use_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_computer_use(d)
  tibble(model = "gpt6_computer_use", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p)
})

alpha_gptluna_computer_use_groups <- gptluna_computer_use_by_play %>% distinct(partner_type)
alpha_gptluna_computer_use_results <- pmap_dfr(alpha_gptluna_computer_use_groups, function(partner_type) {
  d <- gptluna_computer_use_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_computer_use(d)
  tibble(model = "gptluna_computer_use", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p)
})

alpha_results <- bind_rows(alpha_llm_results, alpha_human_results,
                           alpha_computer_use_results, alpha_gpt6_computer_use_results,
                           alpha_gptluna_computer_use_results)


combined_slopes <- inner_join(
  success_results, alpha_results,
  by = c("model", "partner_type")
) %>%
  filter(!is.na(success_slope), !is.na(alpha_slope))



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

cua_raw_names <- c("computer_use", "gpt6_computer_use", "gptluna_computer_use")

plot_df <- combined_slopes %>%
  mutate(
    partner_type = recode(partner_type, !!!player_type_labels),
    model = as.character(model),
    source = case_when(
      model == "Human"         ~ "Human",
      model %in% cua_raw_names ~ "CUA",
      TRUE                     ~ "LLM"
    )
  )


llm_levels_no_human <- sort(unique(plot_df$model[plot_df$source == "LLM"]))

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


find_one <- function(pat) {
  hit <- grep(pat, names(shape_values), ignore.case = TRUE, value = TRUE)
  if (length(hit) != 1) stop("Pattern '", pat, "' matched ", length(hit), " models: ",
                             paste(hit, collapse = ", "))
  hit
}

cua_map <- c(
  "computer_use"         = find_one("gemini"),
  "gpt6_computer_use"    = find_one("gpt-6.*astra"),
  "gptluna_computer_use" = find_one("gpt-5\\.6")
)
print(cua_map)

plot_df <- plot_df %>%
  mutate(model = ifelse(model %in% names(cua_map), unname(cua_map[model]), model))


clean_model_name <- function(x) {
  x <- gsub("gpt", "GPT", x, ignore.case = TRUE)
  x <- gsub("[-_]instruct", "", x, ignore.case = TRUE)
  x
}

names(shape_values) <- clean_model_name(names(shape_values))
plot_df <- plot_df %>% mutate(model = clean_model_name(model))

if (anyDuplicated(names(shape_values))) {
  stop("Name collision after cleaning: ",
       paste(names(shape_values)[duplicated(names(shape_values))], collapse = ", "))
}

shape_values_all <- c("Human" = 15, shape_values)

plot_df <- plot_df %>% mutate(model = factor(model, levels = names(shape_values_all)))
stopifnot(!any(is.na(plot_df$model)))

solid_pat <- "gemini|gpt-5\\.6|gpt-6.*astra"
plot_df <- plot_df %>%
  mutate(fill_color = ifelse(
    model == "Human" | grepl(solid_pat, model, ignore.case = TRUE),
    as.character(partner_type), NA_character_
  ))


x_range <- range(c(0, plot_df$alpha_slope), na.rm = TRUE)
y_range <- range(c(0, plot_df$success_slope), na.rm = TRUE)

make_panel <- function(d, title, y_lab = NULL, point_size = 4,
                       hide_y_text = TRUE, solid_stroke = 0.1,
                       annotate_label = FALSE) {
  d_hollow <- d %>% filter(is.na(fill_color))
  d_solid  <- d %>% filter(!is.na(fill_color))
  
  p <- ggplot() +
  
    annotate("rect", xmin = -Inf, xmax = 0, ymin = 0, ymax = Inf,
             fill = "#FFF3B0", alpha = 0.45) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4)
  

  if (annotate_label) {
    y_lab_pos   <- y_range[2] * 0.80    
    x_arrow_end <- x_range[1] * 0.5      
    x_text      <- x_range[2] * 0.12     
    p <- p +
      annotate("segment",
               x = x_text - x_range[2] * 0.02, y = y_lab_pos,
               xend = x_arrow_end, yend = y_lab_pos,
               arrow = arrow(length = unit(0.22, "cm"), type = "closed"),
               color = "grey35", linewidth = 0.6) +
      annotate("text",
               x = x_text, y = y_lab_pos,
               label = "Consistent with \n Adaptive Behavior",
               hjust = 0, vjust = 0.5, size = 4.5,
               color = "grey25", fontface = "italic", lineheight = 0.9)
  }
  
  p <- p +
    geom_star(
      data = d_hollow,
      aes(x = alpha_slope, y = success_slope, color = partner_type, starshape = model),
      fill = NA, size = point_size, starstroke = 0.8, show.legend = FALSE
    )
  
  if (solid_stroke > 0) {
    p <- p + geom_star(
      data = d_solid,
      aes(x = alpha_slope, y = success_slope, starshape = model, fill = fill_color),
      color = "white", starstroke = solid_stroke, size = point_size, show.legend = FALSE
    )
  } else {
    p <- p + geom_star(
      data = d_solid,
      aes(x = alpha_slope, y = success_slope, color = partner_type,
          starshape = model, fill = fill_color),
      size = point_size, show.legend = FALSE
    )
  }
  
  p <- p +
    labs(title = title, x = "NormalizedTime  Hesitation Time (Tempo) slope", y = y_lab) +
    scale_color_manual(values = condition_colors, guide = "none") +
    scale_fill_manual(values = condition_colors, na.value = NA, guide = "none") +
    scale_starshape_manual(values = shape_values_all, drop = FALSE, guide = "none") +
    scale_x_continuous(limits = x_range) +
    scale_y_continuous(limits = y_range) +
    theme_bw(base_size = 18) +
    theme(legend.position = "none",
          plot.title = element_text(size = 18))
  if (hide_y_text) p <- p + theme(axis.text.y = element_blank())
  p
}

p_human <- make_panel(
  plot_df %>% filter(source == "Human"), title = "Human",
  y_lab = "Success log-odds slope",
  point_size = 5, hide_y_text = FALSE, annotate_label = TRUE
)
p_cua <- make_panel(plot_df %>% filter(source == "CUA"), title = "CUA", point_size = 5)
p_llm <- make_panel(plot_df %>% filter(source == "LLM"), title = "LLM", point_size = 5)


llm_is_solid <- names(shape_values_all) == "Human" |
  grepl(solid_pat, names(shape_values_all), ignore.case = TRUE)
llm_fill_override <- ifelse(llm_is_solid, "black", NA)

shape_legend_plot <- ggplot(plot_df, aes(x = alpha_slope, y = success_slope,
                                         starshape = model, fill = fill_color)) +
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
  theme(legend.position = "right",
        legend.justification = "top",
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 15))

partner_legend_plot <- ggplot(plot_df, aes(x = alpha_slope, y = success_slope,
                                           color = partner_type)) +
  geom_point(alpha = 0, size = 0) +
  scale_color_manual(values = condition_colors, name = "Partner type") +
  guides(color = guide_legend(override.aes = list(alpha = 1, size = 3))) +
  theme_bw(base_size = 18) +
  theme(legend.position = "right",
        legend.justification = "top",
        legend.text = element_text(size = 14, margin = margin(t = 4, b = 4)),
        legend.title = element_text(size = 15))

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

combined <- p_human + p_cua + p_llm + wrap_elements(full = legends_two_col) +
  plot_layout(
    axis_titles = "collect",
    widths = unit(c(1, 1, 1, w_shape + gap + w_partner),
                  c("null", "null", "null", "cm"))
  )

print(combined)

ggsave(
  file.path(ANALYSIS_DIR, "alpha_vs_success_slope_combined.pdf"),
  combined, width = 14.64, height = 4.27, dpi = 300, device = cairo_pdf
)