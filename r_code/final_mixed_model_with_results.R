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
    return(tibble(success_slope = NA_real_, success_p = NA_real_, model_fit = list(NULL)))
  }
  tryCatch({
    formula_str <- if (n_prompt >= 2) {
      "cbind(success, total - success) ~ decile + (1 | prompt_type_re) + (1 | hand_id)"
    } else {
      "cbind(success, total - success) ~ decile + (1 | hand_id)"
    }
    model <- suppressWarnings(glmer(as.formula(formula_str), data = d, family = binomial))
    out <- broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "decile") %>%
      transmute(success_slope = estimate, success_p = p.value)
    out$model_fit <- list(model)
    out
  }, error = function(e) tibble(success_slope = NA_real_, success_p = NA_real_, model_fit = list(NULL)))
}

fit_success_human <- function(d) {
  d <- d %>% filter(total > 0)
  if (nrow(d) == 0 || n_distinct(d$participant) < 2 || n_distinct(d$decile) < 2) {
    return(tibble(success_slope = NA_real_, success_p = NA_real_, model_fit = list(NULL)))
  }
  tryCatch({
    model <- suppressWarnings(
      glmer(cbind(success, total - success) ~ decile + (1 | participant) + (1 | hand_id),
            data = d, family = binomial)
    )
    out <- broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "decile") %>%
      transmute(success_slope = estimate, success_p = p.value)
    out$model_fit <- list(model)
    out
  }, error = function(e) tibble(success_slope = NA_real_, success_p = NA_real_, model_fit = list(NULL)))
}


fit_success_computer_use <- function(d) {
  d <- d %>% filter(total > 0)
  if (nrow(d) == 0 || n_distinct(d$hand_id) < 2 || n_distinct(d$decile) < 2) {
    return(tibble(success_slope = NA_real_, success_p = NA_real_, model_fit = list(NULL)))
  }
  tryCatch({
    model <- suppressWarnings(
      glmer(cbind(success, total - success) ~ decile + (1 | hand_id),
            data = d, family = binomial)
    )
    out <- broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "decile") %>%
      transmute(success_slope = estimate, success_p = p.value)
    out$model_fit <- list(model)
    out
  }, error = function(e) tibble(success_slope = NA_real_, success_p = NA_real_, model_fit = list(NULL)))
}


success_llm_groups <- llm_decile %>% distinct(model, partner_type)
success_llm_results <- pmap_dfr(success_llm_groups, function(model, partner_type) {
  d <- decile_data %>% filter(model == !!model, partner_type == !!partner_type)
  fit <- fit_success_llm_pooled(d)
  tibble(model = model, partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p,
         model_fit = fit$model_fit)
})

success_human_groups <- human_decile %>% distinct(partner_type)
success_human_results <- pmap_dfr(success_human_groups, function(partner_type) {
  d <- human_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_human(d)
  tibble(model = "Human", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p,
         model_fit = fit$model_fit)
})

success_computer_use_groups <- computer_use_decile %>% distinct(partner_type)
success_computer_use_results <- pmap_dfr(success_computer_use_groups, function(partner_type) {
  d <- computer_use_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_computer_use(d)
  tibble(model = "computer_use", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p,
         model_fit = fit$model_fit)
})

success_gpt6_computer_use_groups <- gpt6_computer_use_decile %>% distinct(partner_type)
success_gpt6_computer_use_results <- pmap_dfr(success_gpt6_computer_use_groups, function(partner_type) {
  d <- gpt6_computer_use_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_computer_use(d)
  tibble(model = "gpt6_computer_use", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p,
         model_fit = fit$model_fit)
})

success_gptluna_computer_use_groups <- gptluna_computer_use_decile %>% distinct(partner_type)
success_gptluna_computer_use_results <- pmap_dfr(success_gptluna_computer_use_groups, function(partner_type) {
  d <- gptluna_computer_use_decile %>% filter(partner_type == !!partner_type)
  fit <- fit_success_computer_use(d)
  tibble(model = "gptluna_computer_use", partner_type = partner_type,
         success_slope = fit$success_slope, success_p = fit$success_p,
         model_fit = fit$model_fit)
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
    return(tibble(alpha_slope = NA_real_, alpha_p = NA_real_, model_fit = list(NULL)))
  }
  tryCatch({
    formula_str <- if (n_prompt >= 2) {
      "alpha ~ play_fraction + (1 | prompt_type_re) + (1 | hand_id)"
    } else {
      "alpha ~ play_fraction + (1 | hand_id)"
    }
    model <- suppressWarnings(lmer(as.formula(formula_str), data = d, REML = FALSE))
    out <- broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "play_fraction") %>%
      transmute(alpha_slope = estimate, alpha_p = p.value)
    out$model_fit <- list(model)
    out
  }, error = function(e) tibble(alpha_slope = NA_real_, alpha_p = NA_real_, model_fit = list(NULL)))
}

fit_alpha_human <- function(d) {
  d <- d %>% filter(!is.na(alpha))
  if (nrow(d) == 0 || n_distinct(d$participant) < 2 || n_distinct(d$play_fraction) < 2) {
    return(tibble(alpha_slope = NA_real_, alpha_p = NA_real_, model_fit = list(NULL)))
  }
  tryCatch({
    model <- suppressWarnings(
      lmer(alpha ~ play_fraction + (1 | participant) + (1 | hand_id), data = d, REML = FALSE)
    )
    out <- broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "play_fraction") %>%
      transmute(alpha_slope = estimate, alpha_p = p.value)
    out$model_fit <- list(model)
    out
  }, error = function(e) tibble(alpha_slope = NA_real_, alpha_p = NA_real_, model_fit = list(NULL)))
}


fit_alpha_computer_use <- function(d) {
  d <- d %>% filter(!is.na(alpha))
  if (nrow(d) == 0 || n_distinct(d$hand_id) < 2 || n_distinct(d$play_fraction) < 2) {
    return(tibble(alpha_slope = NA_real_, alpha_p = NA_real_, model_fit = list(NULL)))
  }
  tryCatch({
    model <- suppressWarnings(
      lmer(alpha ~ play_fraction + (1 | hand_id), data = d, REML = FALSE)
    )
    out <- broom.mixed::tidy(model, effects = "fixed") %>%
      filter(term == "play_fraction") %>%
      transmute(alpha_slope = estimate, alpha_p = p.value)
    out$model_fit <- list(model)
    out
  }, error = function(e) tibble(alpha_slope = NA_real_, alpha_p = NA_real_, model_fit = list(NULL)))
}

alpha_llm_groups <- llm_by_play %>% distinct(model, partner_type)
alpha_llm_results <- pmap_dfr(alpha_llm_groups, function(model, partner_type) {
  d <- llm_by_play %>% filter(model == !!model, partner_type == !!partner_type)
  fit <- fit_alpha_llm_pooled(d)
  tibble(model = model, partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p,
         model_fit = fit$model_fit)
})

alpha_human_groups <- human_by_play %>% distinct(partner_type)
alpha_human_results <- pmap_dfr(alpha_human_groups, function(partner_type) {
  d <- human_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_human(d)
  tibble(model = "Human", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p,
         model_fit = fit$model_fit)
})

alpha_computer_use_groups <- computer_use_by_play %>% distinct(partner_type)
alpha_computer_use_results <- pmap_dfr(alpha_computer_use_groups, function(partner_type) {
  d <- computer_use_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_computer_use(d)
  tibble(model = "computer_use", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p,
         model_fit = fit$model_fit)
})

alpha_gpt6_computer_use_groups <- gpt6_computer_use_by_play %>% distinct(partner_type)
alpha_gpt6_computer_use_results <- pmap_dfr(alpha_gpt6_computer_use_groups, function(partner_type) {
  d <- gpt6_computer_use_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_computer_use(d)
  tibble(model = "gpt6_computer_use", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p,
         model_fit = fit$model_fit)
})

alpha_gptluna_computer_use_groups <- gptluna_computer_use_by_play %>% distinct(partner_type)
alpha_gptluna_computer_use_results <- pmap_dfr(alpha_gptluna_computer_use_groups, function(partner_type) {
  d <- gptluna_computer_use_by_play %>% filter(partner_type == !!partner_type)
  fit <- fit_alpha_computer_use(d)
  tibble(model = "gptluna_computer_use", partner_type = partner_type,
         alpha_slope = fit$alpha_slope, alpha_p = fit$alpha_p,
         model_fit = fit$model_fit)
})

alpha_results <- bind_rows(alpha_llm_results, alpha_human_results,
                           alpha_computer_use_results, alpha_gpt6_computer_use_results,
                           alpha_gptluna_computer_use_results)


print_model_summary <- function(results, model_name, ptype) {
  row <- results %>% filter(model == model_name, partner_type == ptype)
  if (nrow(row) == 0) {
    message("没找到 model = ", model_name, ", partner_type = ", ptype, " 这个组合")
    return(invisible(NULL))
  }
  fit <- row$model_fit[[1]]
  if (is.null(fit)) {
    message("这个组合的模型没拟合成功（数据不够 / 报错），没有 summary 可看")
    return(invisible(NULL))
  }
  print(summary(fit))
}

dump_all_model_summaries <- function(results, path, label = "") {
  con <- file(path, open = "wt")
  on.exit(close(con))
  for (i in seq_len(nrow(results))) {
    cat(
      "\n\n================================================================\n",
      label, " | model = ", results$model[i], " | partner_type = ", results$partner_type[i], "\n",
      "================================================================\n",
      sep = "", file = con
    )
    fit <- results$model_fit[[i]]
    if (is.null(fit)) {
      cat("(模型没拟合成功，跳过)\n", file = con)
    } else {
      capture.output(print(summary(fit)), file = con, append = TRUE)
    }
  }
  message("已写入 ", path)
}


print_model_summary(success_results, "Human", "bayesian_true")
print_model_summary(alpha_results, "GPT6_computer_use", "random")
dump_all_model_summaries(success_results, file.path(ANALYSIS_DIR, "success_model_summaries.txt"), label = "SUCCESS")
dump_all_model_summaries(alpha_results,   file.path(ANALYSIS_DIR, "alpha_model_summaries.txt"),   label = "ALPHA")


combined_slopes <- inner_join(
  success_results %>% select(model, partner_type, success_slope, success_p),
  alpha_results %>% select(model, partner_type, alpha_slope, alpha_p),
  by = c("model", "partner_type")
) %>%
  filter(!is.na(success_slope), !is.na(alpha_slope))

write.csv(
  combined_slopes,
  file.path(ANALYSIS_DIR, "alpha_vs_success_slope_pooled.csv"),
  row.names = FALSE
)


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




combined_slopes <- combined_slopes %>%
  mutate(partner_type = recode(partner_type, !!!player_type_labels))


llm_model_names <- setdiff(sort(unique(combined_slopes$model)),
                           c("Human", "computer_use", "gpt6_computer_use", "gptluna_computer_use"))


model_shape_patterns <- list(
  list(pattern = "gemini",             starshape = 13),
  list(pattern = "gpt-5\\.6",          starshape = 5),
  list(pattern = "gpt-oss.*120b",      starshape = 15),
  list(pattern = "gpt-oss.*[^0-9]20b", starshape = 11),
  list(pattern = "llama.*70b",         starshape = 23),
  list(pattern = "llama.*[^0-9]8b",    starshape = 12),
  list(pattern = "llama.*[^0-9]1b",    starshape = 6),
  list(pattern = "qwen.*32b",          starshape = 13),
  list(pattern = "qwen.*[^0-9]8b",     starshape = 5),
  list(pattern = "gpt-6",              starshape = 6)
)

starshape_values <- setNames(rep(NA_real_, length(llm_model_names)), llm_model_names)
for (p in model_shape_patterns) {
  idx <- grepl(p$pattern, names(starshape_values), ignore.case = TRUE)
  starshape_values[idx] <- p$starshape
}
if (any(is.na(starshape_values))) {
  warning("Some LLM models do not have a starshape assigned: ",
          paste(names(starshape_values)[is.na(starshape_values)], collapse = ", "))
}

starshape_values <- c("Human" = 11, starshape_values)


combined_slopes <- combined_slopes %>%
  mutate(model = recode(as.character(model),
                        "computer_use" = "Gemini (CUA)",
                        "gpt6_computer_use" = "GPT-6 (CUA)",
                        "gptluna_computer_use" = "GPT-5.6-Luna (CUA)"))

starshape_values <- c(
  starshape_values["Human"],
  c("Gemini (CUA)" = 23, "GPT-6 (CUA)" = 15, "GPT-5.6-Luna (CUA)" = 12),
  starshape_values[names(starshape_values) != "Human"]
)

combined_slopes <- combined_slopes %>%
  mutate(model = factor(model, levels = names(starshape_values)))


combined_slopes <- combined_slopes %>%
  mutate(fill_color = ifelse(
    model %in% c("Human", "gemini-3.7-flash", "gpt-5.6-luna",
                 "Gemini (CUA)", "GPT-6 (CUA)", "GPT-5.6-Luna (CUA)", "gpt-6-astra"),
    as.character(partner_type), NA_character_
  ))


combined_slopes_star <- combined_slopes

scatter_plot <- ggplot(
  combined_slopes,
  aes(x = alpha_slope, y = success_slope, color = partner_type)
) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_star(
    data = combined_slopes_star,
    aes(starshape = model, fill = fill_color),
    size = 4, alpha = 0.85, starstroke = 0.8,
    show.legend = c(colour = FALSE, starshape = TRUE, fill = FALSE)
  ) +

  geom_point(aes(x = alpha_slope, y = success_slope, color = partner_type),
             alpha = 0, size = 0) +
  scale_color_manual(values = condition_colors) +
  scale_fill_manual(values = condition_colors, na.value = NA, guide = "none") +
  scale_starshape_manual(values = starshape_values) +
  guides(
    starshape = guide_legend(override.aes = list(
      size = 5, alpha = 1,
      fill = ifelse(names(starshape_values) %in%
                      c("Human", "gemini-3.7-flash", "gpt-5.6-luna",
                        "Gemini (CUA)", "GPT-6 (CUA)", "GPT-5.6-Luna (CUA)", 'gpt-6-astra'),
                    "grey30", NA)
    )),

    color = guide_legend(override.aes = list(size = 4, alpha = 1))
  ) +
  labs(
    x = "Speed slope \n Negative indicates convention formation",
    y = "Success log-odds slope \n Positive indicates convention formation",
    color = "Partner type",
    starshape = "Player type"
  ) +
  theme_bw(base_size = 18) +
  theme(
    legend.position = "right",
    legend.text = element_text(size = 15),
    legend.title = element_text(size = 18),
    legend.spacing.y = unit(0.1, "cm"),
    legend.box.spacing = unit(0.2, "cm"),
    legend.margin = margin(0, 0, 0, 0),
    # axis.text.y = element_blank()
  )

print(scatter_plot)

ggsave(
  file.path(ANALYSIS_DIR, "alpha_vs_success_slope.pdf"),
  scatter_plot,
  width = 7.4, height = 5, dpi = 300
)