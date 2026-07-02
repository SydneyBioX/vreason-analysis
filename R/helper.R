MODEL_NAMES <- c(
  gemma_3_4b_it = "Gemma-3 4B",
  hulu_med_7b = "HuLu-Med 7B",
  internvl3_8b = "InternVL3 8B", # TODO: this is a type. We evaluated 3.5 not 3. 
  internvl3_5_8b = "InternVL3.5 8B",
  llama_3.2_11b_vision_instruct = "LLaMA-3.2 11B",
  `llava-med` = "LLaVA-Med",
  medgemma_4b_it = "MedGemma 4B",
  phi_3.5_vision_instruct = "Phi-3.5 Vision",
  qwen2.5_vl_7b_instruct = "Qwen2.5-VL 7B",
  qwen2.5_vl_7b_instruct_normal = "Qwen2.5-VL 7B",
  qwen2.5_vl_7b_instruct_cot = "Qwen2.5-VL 7B (CoT)",
  qwen3_vl_8b_instruct = "Qwen3-VL 8B",
  qwen3_vl_8b_instruct_cot = "Qwen3-VL 8B (CoT)",
  vreasson = "VReason", # TODO: fix this random type in the evaluation code.
  vreason = "VReason"
)

SKIP_STEMS <- c("hulu_med_7b", "medgemma_4b_it")

should_skip <- function(stem) {
  stem %in% SKIP_STEMS | grepl("cot", stem, ignore.case = TRUE)
}

model_label <- function(stem) {
  lbl <- MODEL_NAMES[stem]
  ifelse(is.na(lbl), stem, lbl)
}

MODEL_PALETTE <- c(
  "Gemma-3 4B" = "#E05780",
  "InternVL3 8B" = "#F0B84D", # fix for the original model naming error.
  "InternVL3.5 8B" = "#F0B84D",
  "LLaMA-3.2 11B" = "#3B82F6",
  "LLaVA-Med" = "#7CB518",
  "Phi-3.5 Vision" = "#2EC4B6",
  "Qwen2.5-VL 7B" = "#C9A0DC",
  "Qwen3-VL 8B" = "#F4845F",
  "VReason" = "#8B1A1A"
)

get_model_colours <- function(models) {
  cols <- MODEL_PALETTE[models]
  missing <- is.na(cols)
  if (any(missing)) {
    extra <- hue_pal()(sum(missing))
    cols[missing] <- extra
  }
  names(cols) <- models
  cols
}

# Data loading 
load_per_sample <- function(directory) {
  csvs <- list.files(directory, pattern = "\\.csv$", full.names = TRUE)
  csvs <- csvs[!grepl("summary\\.csv$", csvs)]
  frames <- map(csvs, function(p) {
    stem <- tools::file_path_sans_ext(basename(p))
    if (should_skip(stem)) return(NULL)
    df <- read_csv(p, show_col_types = FALSE)
    df$model       <- stem
    df$model_label <- model_label(stem)
    df
  })
  bind_rows(compact(frames))
}

# Bootstrap CI
boot_mean_ci <- function(x, n_boot = N_BOOT, seed = SEED) {
  x <- x[!is.na(x)]
  n <- length(x)
  set.seed(seed)
  
  boot_means <- replicate(n_boot, mean(sample(x, n, replace = TRUE)))
  tibble(
    mean  = mean(x),
    ci_lo = quantile(boot_means, 0.025),
    ci_hi = quantile(boot_means, 0.975)
  )
}

compute_boot_cis <- function(ps_df, metrics) {
  ps_df |>
    select(model_label, all_of(intersect(metrics, names(ps_df)))) |>
    pivot_longer(-model_label, names_to = "metric", values_to = "value") |>
    group_by(model_label, metric) |>
    summarise(boot_mean_ci(value), .groups = "drop")
}

build_summary <- function(ps_df, metrics) {
  ps_df |>
    group_by(model, model_label) |>
    summarise(
      across(all_of(intersect(metrics, names(ps_df))), \(x) mean(x, na.rm = TRUE)),
      n = n(), .groups = "drop"
    )
}

# Plotting 
save_plot <- function(p, ..., width = 10, height = 6) {
  fname <- paste0(paste(..., sep = "_"), ".pdf")
  ggsave(file.path(FIG_DIR, fname), plot = p, width = width, height = height,
         device = "pdf")
}

bar_with_ci <- function(boot_df, metrics, title = "", ylim_max = NA) {
  plot_df <- boot_df |>
    filter(metric %in% metrics) |>
    mutate(metric = factor(metric, levels = metrics))
  models <- unique(plot_df$model_label)
  cols <- get_model_colours(models)
  y_lab <- if (length(metrics) == 1) str_replace_all(metrics, "_", " ") else "Score"
  
  p <- ggplot(plot_df, aes(x = model_label, y = mean, fill = model_label)) +
    geom_col(width = 0.7, alpha = 0.85) +
    geom_errorbar(
      aes(ymin = ci_lo, ymax = ci_hi),
      width = 0.25, linewidth = 0.4
    ) +
    scale_fill_manual(values = cols) +
    scale_x_discrete(labels = \(x) str_wrap(x, width = 12)) +
    labs(title = title, x = NULL, y = y_lab, fill = NULL) +
    coord_cartesian(ylim = c(0, ylim_max)) +
    theme_bw(base_size = 10) +
    theme(
      panel.grid.major.x = element_blank(),
      axis.text.x  = element_text(size = 8),
      legend.position = "none",
      plot.title   = element_text(face = "bold", size = 11)
    )
  
  if (n_distinct(plot_df$metric) > 1) {
    p <- p +
      facet_wrap(~ metric, scales = "free_y",
                 labeller = labeller(metric = \(x) str_replace_all(x, "_", " "))) +
      labs(y = "Score")
  }
  p
}

# hard-code the limits for the radar plots (purely aesthetic).
MIMIC_CXR_RADAR_LIMITS <- list(
  bleu1 = c(0.10, 0.32),
  bleu2 = c(0.04, 0.17),
  bleu4 = c(0.00, 0.06),
  meteor = c(0.15, 0.27),
  rougel = c(0.10, 0.20),
  cider = c(0.00, 0.10),
  bertscore = c(0.84, 0.90),
  semb_score = c(0.20, 0.40),
  radgraph_combined = c(0.10, 0.25),
  CE_precision = c(0.70, 0.80),
  CE_recall = c(0.70, 0.80),
  CE_f1 = c(0.70, 0.80),
  CE5_precision = c(0.60, 0.75),
  CE5_recall = c(0.60, 0.75),
  CE5_f1 = c(0.60, 0.75)
)

axis_limit_df <- function(metrics, axis_limits, max_value) {
  if (is.null(axis_limits)) {
    return(tibble(
      metric = metrics,
      axis_min = 0,
      axis_max = max_value
    ))
  }
  
  missing_limits <- setdiff(metrics, names(axis_limits))
  if (length(missing_limits) > 0) {
    stop(
      "Missing radar axis limits for: ",
      paste(missing_limits, collapse = ", "),
      call. = FALSE
    )
  }
  
  limits_df <- tibble(metric = metrics) |>
    mutate(
      axis_min = map_dbl(metric, \(m) axis_limits[[m]][1]),
      axis_max = map_dbl(metric, \(m) axis_limits[[m]][2])
    )
  bad <- limits_df |>
    filter(!is.finite(axis_min) | !is.finite(axis_max) | axis_max <= axis_min)
  if (nrow(bad) > 0) {
    stop(
      "Invalid radar axis limits for: ",
      paste(bad$metric, collapse = ", "),
      call. = FALSE
    )
  }
  limits_df
}

axis_value_label <- function(x) {
  number(x, accuracy = 0.01)
}

radar_plot <- function(summary_df, metrics, title = "", max_value = 1,
                       axis_limits = NULL,
                       label_size = 3.2, grid_label_size = 2.8,
                       fill_alpha = 0.2, line_width = 1.1,
                       point_size = 1.8) {
  avail <- intersect(metrics, names(summary_df))
  if (length(avail) < 3) return(ggplot() + theme_void())
  use_axis_limits <- !is.null(axis_limits)
  
  metric_labels <- c(
    bleu1 = "BLEU-1",
    bleu2 = "BLEU-2",
    bleu3 = "BLEU-3",
    bleu4 = "BLEU-4",
    meteor = "METEOR",
    rougel = "ROUGE-L",
    cider = "CIDEr",
    bertscore = "BERTScore",
    semb_score = "SEMB",
    radgraph_combined = "RadGraph F1",
    CE_precision = "CE precision",
    CE_recall = "CE recall",
    CE_f1 = "CE F1",
    CE5_precision = "CE5 precision",
    CE5_recall = "CE5 recall",
    CE5_f1 = "CE5 F1",
    exact_match = "Exact match",
    contains_match = "Contains match",
    token_recall = "Token recall",
    token_f1 = "Token F1"
  )
  
  metric_lookup <- tibble(
    metric = avail,
    metric_idx = seq_along(avail),
    theta = seq(0, 2 * pi, length.out = length(avail) + 1)[-(length(avail) + 1)]
  ) |>
    left_join(axis_limit_df(avail, axis_limits, max_value), by = "metric")
  
  plot_df <- summary_df |>
    select(model_label, all_of(avail)) |>
    pivot_longer(-model_label, names_to = "metric", values_to = "value") |>
    left_join(metric_lookup, by = "metric") |>
    mutate(
      metric = factor(metric, levels = avail),
      radius = (value - axis_min) / (axis_max - axis_min),
      x = radius * sin(theta),
      y = radius * cos(theta)
    ) |>
    arrange(model_label, metric_idx)
  
  out_of_range <- plot_df |>
    filter(!is.na(value), value < axis_min | value > axis_max)
  if (nrow(out_of_range) > 0) {
    offenders <- out_of_range |>
      transmute(
        label = paste0(
          model_label, " ", as.character(metric), "=",
          axis_value_label(value), " outside [",
          axis_value_label(axis_min), ", ",
          axis_value_label(axis_max), "]"
        )
      ) |>
      pull(label)
  }
  
  cols <- get_model_colours(unique(plot_df$model_label))
  
  path_df <- plot_df |>
    group_by(model_label) |>
    group_modify(~ bind_rows(.x, slice(.x, 1))) |>
    ungroup()
  
  grid_breaks <- if (use_axis_limits) seq(0.25, 1, by = 0.25) else seq(0.25, max_value, by = 0.25)
  grid_angles <- seq(0, 2 * pi, length.out = 361)
  grid_df <- crossing(
    radius = if (use_axis_limits) grid_breaks else grid_breaks / max_value,
    theta = grid_angles
  ) |>
    mutate(
      x = radius * sin(theta),
      y = radius * cos(theta)
    )
  
  axis_df <- metric_lookup |>
    mutate(
      xend = sin(theta),
      yend = cos(theta)
    )
  
  label_df <- metric_lookup |>
    mutate(
      metric_label = str_wrap(
        recode(metric, !!!metric_labels, .default = metric),
        width = if (use_axis_limits) 10 else 12
      ),
      axis_label = if (use_axis_limits) {
        paste0("[", axis_value_label(axis_min), "-", axis_value_label(axis_max), "]")
      } else {
        rep("", n())
      },
      label = if (use_axis_limits) paste(metric_label, axis_label, sep = "\n") else metric_label,
      x = 1.1 * sin(theta),
      y = 1.1 * cos(theta),
      hjust = case_when(x < -0.05 ~ 1, x > 0.05 ~ 0, TRUE ~ 0.5),
      vjust = case_when(y < -0.05 ~ 1, y > 0.05 ~ 0, TRUE ~ 0.5)
    )
  
  grid_label_df <- if (use_axis_limits) {
    tibble(radius = numeric(), label = character(), x = numeric(), y = numeric())
  } else {
    tibble(
      radius = grid_breaks / max_value,
      label = number(grid_breaks, accuracy = 0.01),
      x = 0.035,
      y = radius
    )
  }
  
  axis_tick_df <- if (use_axis_limits) {
    metric_lookup |>
      mutate(
        radius = 0.5,
        value = (axis_min + axis_max) / 2,
        label = axis_value_label(value),
        x = radius * sin(theta),
        y = radius * cos(theta),
        tick_dx = 0.025 * cos(theta),
        tick_dy = -0.025 * sin(theta),
        hjust = case_when(x < -0.05 ~ 1, x > 0.05 ~ 0, TRUE ~ 0.5),
        vjust = case_when(y < -0.05 ~ 1, y > 0.05 ~ 0, TRUE ~ 0.5)
      )
  } else {
    tibble(
      radius = numeric(), value = numeric(), label = character(),
      x = numeric(), y = numeric(), tick_dx = numeric(), tick_dy = numeric(),
      hjust = numeric(), vjust = numeric()
    )
  }
  
  p <- ggplot() +
    geom_path(
      data = grid_df,
      aes(x = x, y = y, group = radius),
      colour = "grey88",
      linewidth = 0.45
    ) +
    geom_segment(
      data = axis_df,
      aes(x = 0, y = 0, xend = xend, yend = yend),
      colour = "grey88",
      linewidth = 0.45
    ) +
    geom_polygon(
      data = path_df,
      aes(x = x, y = y, group = model_label, fill = model_label),
      alpha = fill_alpha,
      colour = NA,
      show.legend = FALSE
    ) +
    geom_path(
      data = path_df,
      aes(x = x, y = y, group = model_label, colour = model_label),
      linewidth = line_width,
      linejoin = "round",
      lineend = "round",
      show.legend = TRUE
    ) +
    geom_point(
      data = plot_df,
      aes(x = x, y = y, colour = model_label),
      size = point_size,
      show.legend = FALSE
    ) +
    geom_text(
      data = label_df,
      aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
      colour = "grey25",
      size = label_size
    ) +
    geom_text(
      data = grid_label_df,
      aes(x = x, y = y, label = label),
      colour = "grey45",
      size = grid_label_size,
      hjust = 0
    )
  
  if (use_axis_limits) {
    p <- p +
      geom_segment(
        data = axis_tick_df,
        aes(x = x - tick_dx, y = y - tick_dy, xend = x + tick_dx, yend = y + tick_dy),
        colour = "grey62",
        linewidth = 0.35
      ) +
      geom_text(
        data = axis_tick_df,
        aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
        colour = "grey35",
        size = grid_label_size,
        nudge_x = 0.035,
        nudge_y = 0.02
      )
  }
  
  p +
    scale_colour_manual(values = cols) +
    scale_fill_manual(values = cols) +
    guides(fill = "none") +
    coord_equal(xlim = c(-1.52, 1.52), ylim = c(-1.32, 1.32), expand = FALSE, clip = "off") +
    labs(title = title, colour = NULL, fill = NULL) +
    theme_void(base_size = 10) +
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = 8),
      plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
      plot.margin = margin(12, 34, 12, 34)
    )
}

metric_heatmap <- function(summary_df, content_types, metric, title = "") {
  cols <- paste0("content_type_", content_types, "_", metric)
  avail_idx <- cols %in% names(summary_df)
  if (sum(avail_idx) == 0) return(ggplot() + theme_void())
  
  summary_df |>
    select(model_label, all_of(cols[avail_idx])) |>
    pivot_longer(-model_label, names_to = "content_type", values_to = "value") |>
    mutate(content_type = str_replace(content_type,
                                      paste0("content_type_(.+)_", metric), "\\1")) |>
    ggplot(aes(x = content_type, y = model_label, fill = value)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
    scale_fill_gradient(low = "white", high = "#8B1A1A", limits = c(0, 1)) +
    labs(title = title, x = NULL, y = NULL, fill = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1, size = 8),
      plot.title  = element_text(face = "bold", size = 11)
    )
}
