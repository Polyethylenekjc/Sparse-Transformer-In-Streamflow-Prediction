#!/usr/bin/env Rscript

required_pkgs <- c("readr", "dplyr", "tidyr", "ggplot2", "patchwork", "maps", "jsonlite")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    sprintf(
      "Missing R packages: %s\nInstall with:\ninstall.packages(c(%s))",
      paste(missing_pkgs, collapse = ", "),
      paste(sprintf('"%s"', missing_pkgs), collapse = ", ")
    )
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(maps)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    rf_csv = "outputs/random_forest_model_performance.csv",
    stations_csv = "data/stations.csv",
    sparse_output_root = "",
    sparse_nse_threshold = 0.5,
    out_figure = "outputs/figures/Figure_S_RF_Performance_4x2.svg",
    out_subset_csv = "outputs/figures/Figure_S_RF_Performance_4x2_subset.csv",
    width = 13,
    height = 16,
    dpi = 320
  )

  if (length(args) == 0) {
    return(out)
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--rf-csv" && i + 1 <= length(args)) {
      out$rf_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--stations-csv" && i + 1 <= length(args)) {
      out$stations_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--sparse-output-root" && i + 1 <= length(args)) {
      out$sparse_output_root <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--sparse-nse-threshold" && i + 1 <= length(args)) {
      out$sparse_nse_threshold <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--out-figure" && i + 1 <= length(args)) {
      out$out_figure <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out-subset-csv" && i + 1 <= length(args)) {
      out$out_subset_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--width" && i + 1 <= length(args)) {
      out$width <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--height" && i + 1 <= length(args)) {
      out$height <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--dpi" && i + 1 <= length(args)) {
      out$dpi <- as.integer(args[[i + 1]])
      i <- i + 2
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }

  out
}

pad_station_id <- function(x) {
  sprintf("%08s", as.character(x))
}

resolve_sparse_root <- function(cli_path) {
  if (!is.null(cli_path) && nchar(trimws(cli_path)) > 0) {
    return(normalizePath(path.expand(cli_path), winslash = "/", mustWork = FALSE))
  }

  preferred <- "/mnt/d/Store/Sparse_Transformer"
  if (dir.exists(preferred)) {
    return(preferred)
  }

  "outputs"
}

load_sparse_good_stations <- function(sparse_root, threshold = 0.5) {
  csv_path <- file.path(sparse_root, "sparse_transformer_station_metrics.csv")

  if (file.exists(csv_path)) {
    df <- readr::read_csv(csv_path, show_col_types = FALSE)
    if (all(c("station_id", "nse") %in% names(df))) {
      sid <- pad_station_id(df$station_id)
      nse <- suppressWarnings(as.numeric(df$nse))
      keep <- is.finite(nse) & (nse > threshold)
      return(unique(sid[keep]))
    }
  }

  dirs <- list.dirs(sparse_root, recursive = FALSE, full.names = TRUE)
  ids <- c()
  for (d in dirs) {
    sid <- basename(d)
    if (!grepl("^[0-9]+$", sid)) {
      next
    }
    js <- file.path(d, "summary.json")
    if (!file.exists(js)) {
      next
    }
    obj <- tryCatch(jsonlite::fromJSON(js), error = function(e) NULL)
    if (is.null(obj) || is.null(obj$metrics) || is.null(obj$metrics$nse)) {
      next
    }
    nse <- suppressWarnings(as.numeric(obj$metrics$nse))
    if (is.finite(nse) && nse > threshold) {
      ids <- c(ids, pad_station_id(sid))
    }
  }

  unique(ids)
}

base_theme <- theme_bw(base_family = "Times New Roman", base_size = 16) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
    axis.title = element_text(face = "bold", size = 16),
    axis.text = element_text(size = 14),
    legend.title = element_text(face = "bold", size = 14),
    legend.text = element_text(size = 13),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.25),
    panel.grid.minor = element_blank()
  )

winsorize_vec <- function(x, p = 0.01) {
  x <- as.numeric(x)
  ok <- is.finite(x)
  if (sum(ok) < 5) {
    return(x)
  }
  lo <- as.numeric(stats::quantile(x[ok], probs = p, na.rm = TRUE))
  hi <- as.numeric(stats::quantile(x[ok], probs = 1 - p, na.rm = TRUE))
  x[x < lo] <- lo
  x[x > hi] <- hi
  x
}

build_perf_panel <- function(df_plot, model = c("Baseline", "Modified")) {
  model <- match.arg(model)
  if (model == "Baseline") {
    df <- df_plot %>%
      transmute(
        station_id,
        NSE = as.numeric(nse_base),
        KGE = as.numeric(kge_base),
        MSE = as.numeric(mse_base),
        MAE = if ("mae_base" %in% names(df_plot)) as.numeric(mae_base) else sqrt(pmax(as.numeric(mse_base), 0)) * sqrt(2 / pi)
      )
    panel_title <- "(b) Baseline model performance distribution"
  } else {
    df <- df_plot %>%
      transmute(
        station_id,
        NSE = as.numeric(nse_mod),
        KGE = as.numeric(kge_mod),
        MSE = as.numeric(mse_mod),
        MAE = if ("mae_mod" %in% names(df_plot)) as.numeric(mae_mod) else sqrt(pmax(as.numeric(mse_mod), 0)) * sqrt(2 / pi)
      )
    panel_title <- "(c) Modified model performance distribution"
  }

  perf_long <- df %>%
    pivot_longer(cols = c("NSE", "KGE", "MSE", "MAE"), names_to = "metric", values_to = "value") %>%
    mutate(
      metric = factor(metric, levels = c("NSE", "KGE", "MSE", "MAE"))
    )

  perf_long <- perf_long %>%
    group_by(metric) %>%
    group_modify(~ {
      m <- as.character(.y$metric[[1]])
      p_clip <- if (m == "KGE") 0.05 else 0.02
      .x$value <- winsorize_vec(.x$value, p = p_clip)
      .x
    }) %>%
    ungroup()

  left_df <- perf_long %>% filter(metric %in% c("NSE", "KGE"), is.finite(value))
  right_df <- perf_long %>% filter(metric %in% c("MSE", "MAE"), is.finite(value))

  lmin <- as.numeric(stats::quantile(left_df$value, probs = 0.01, na.rm = TRUE))
  lmax <- as.numeric(stats::quantile(left_df$value, probs = 0.99, na.rm = TRUE))
  rmin <- as.numeric(stats::quantile(right_df$value, probs = 0.01, na.rm = TRUE))
  rmax <- as.numeric(stats::quantile(right_df$value, probs = 0.99, na.rm = TRUE))

  if (!is.finite(lmin) || !is.finite(lmax) || lmin == lmax) {
    lmin <- -1
    lmax <- 1
  }
  if (!is.finite(rmin) || !is.finite(rmax) || rmin == rmax) {
    rmin <- 0
    rmax <- 1
  }

  to_left <- function(x) {
    (x - rmin) / (rmax - rmin) * (lmax - lmin) + lmin
  }
  from_left <- function(y) {
    (y - lmin) / (lmax - lmin) * (rmax - rmin) + rmin
  }

  right_df <- right_df %>% mutate(y_plot = to_left(value))
  left_df <- left_df %>% mutate(y_plot = value)
  plot_df <- bind_rows(left_df, right_df)

  ggplot(plot_df, aes(x = metric, y = y_plot, fill = metric)) +
    geom_boxplot(width = 0.65, outlier.alpha = 0.22, linewidth = 0.3) +
    scale_fill_manual(
      values = c(
        NSE = "#1b9e77",
        KGE = "#7570b3",
        MSE = "#d95f02",
        MAE = "#e7298a"
      ),
      name = "Performance metric",
      guide = guide_legend(order = 2, nrow = 1)
    ) +
    scale_y_continuous(
      name = "NSE and KGE",
      sec.axis = sec_axis(~ from_left(.), name = "MSE and MAE")
    ) +
    labs(
      title = panel_title,
      x = NULL
    ) +
    base_theme +
    theme(
      axis.title.x = element_text(size = 13, face = "plain"),
      axis.title.y = element_text(size = 13, face = "plain"),
      axis.title.y.right = element_text(size = 13, face = "plain")
    )
}

make_figure <- function(df_plot, usa_map) {
  n_total <- nrow(df_plot)
  n_mod_better <- sum(df_plot$delta_nse > 0, na.rm = TRUE)
  n_base_better <- sum(df_plot$delta_nse < 0, na.rm = TRUE)
  n_tie <- sum(df_plot$delta_nse == 0, na.rm = TRUE)
  map_note <- sprintf(
    "Stations: %d\nBaseline better: %d\nModified better: %d\n",
    n_total,
    n_base_better,
    n_mod_better,
    n_tie
  )

  dabs <- abs(df_plot$delta_nse)
  d_scale <- as.numeric(stats::quantile(dabs[is.finite(dabs)], probs = 0.50, na.rm = TRUE))
  if (!is.finite(d_scale) || d_scale <= 0) {
    d_scale <- 0.02
  }
  df_plot <- df_plot %>%
    mutate(
      delta_nse_vis = asinh(delta_nse / d_scale),
      change_direction = ifelse(delta_nse >= 0, "Improved", "Degraded")
    )

  max_abs <- as.numeric(stats::quantile(abs(df_plot$delta_nse), probs = 0.98, na.rm = TRUE))
  if (!is.finite(max_abs) || max_abs <= 0) {
    max_abs <- max(abs(df_plot$delta_nse), na.rm = TRUE)
  }
  if (!is.finite(max_abs) || max_abs <= 0) {
    max_abs <- 0.1
  }
  tick_raw <- c(-max_abs, 0, max_abs)
  tick_vis <- asinh(tick_raw / d_scale)

  p_map <- ggplot() +
    geom_polygon(
      data = usa_map,
      aes(x = long, y = lat, group = group),
      fill = "grey95",
      color = "grey60",
      linewidth = 0.2
    ) +
    geom_point(
      data = df_plot,
      aes(x = lon, y = lat, color = delta_nse_vis),
      size = 2.7,
      alpha = 0.96
    ) +
    scale_color_gradient2(
      low = "#08519c",
      mid = "#f7f7f7",
      high = "#cb181d",
      midpoint = 0,
      limits = range(tick_vis),
      breaks = tick_vis,
      labels = sprintf("%.2f", tick_raw),
      oob = scales::squish,
      name = expression(Delta * "  NSE"),
      guide = guide_colorbar(
        order = 1,
        direction = "horizontal",
        title.position = "top",
        label.position = "bottom",
        barwidth = grid::unit(8.4, "cm"),
        barheight = grid::unit(0.46, "cm")
      )
    ) +
    coord_quickmap(xlim = c(-125, -66), ylim = c(24, 50), expand = FALSE) +
    labs(
      title = "(a) Spatial difference in NSE between modified and baseline models",
      x = "Longitude",
      y = "Latitude"
    ) +
    annotate(
      "text",
      x = -66.5,
      y = 24.2,
      label = map_note,
      hjust = 1,
      vjust = 0,
      size = 4.2,
      family = "Times New Roman"
    ) +
    base_theme

  p_box_base <- build_perf_panel(df_plot, model = "Baseline")
  p_box_mod <- build_perf_panel(df_plot, model = "Modified")

  dtest <- df_plot %>% filter(is.finite(nse_base), is.finite(nse_mod))
  p_t <- NA_real_
  p_w <- NA_real_
  if (nrow(dtest) >= 5) {
    p_t <- tryCatch(stats::t.test(dtest$nse_mod, dtest$nse_base, paired = TRUE)$p.value, error = function(e) NA_real_)
    p_w <- tryCatch(stats::wilcox.test(dtest$nse_mod, dtest$nse_base, paired = TRUE)$p.value, error = function(e) NA_real_)
  }
  d_note <- sprintf("Paired test: t-test p = %s; Wilcoxon p = %s",
                    ifelse(is.finite(p_t), formatC(p_t, format = "e", digits = 2), "NA"),
                    ifelse(is.finite(p_w), formatC(p_w, format = "e", digits = 2), "NA"))
  d_note <- gsub("; ", "\n", d_note, fixed = TRUE)

  p_scatter_1 <- ggplot(df_plot, aes(x = nse_base, y = nse_mod, color = change_direction)) +
    geom_point(alpha = 0.70, size = 1.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    annotate("text", x = Inf, y = -Inf, label = d_note, hjust = 1.02, vjust = -0.25, size = 4.2, family = "Times New Roman") +
    scale_color_manual(values = c(Improved = "#1b9e77", Degraded = "#d95f02"), name = "Change direction", guide = guide_legend(order = 2, nrow = 1)) +
    labs(
      title = "(d) Station-wise NSE comparison",
      x = "Baseline model NSE",
      y = "Modified model NSE"
    ) +
    base_theme +
    theme(
      axis.title.x = element_text(size = 13, face = "plain"),
      axis.title.y = element_text(size = 13, face = "plain")
    )

  lm_df <- df_plot %>% filter(is.finite(nse_base), is.finite(delta_nse))
  lm_note <- "Linear fit: R2 = NA; p = NA"
  if (nrow(lm_df) >= 5) {
    fit <- stats::lm(delta_nse ~ nse_base, data = lm_df)
    sm <- summary(fit)
    fit_p <- tryCatch(sm$coefficients[2, 4], error = function(e) NA_real_)
    fit_r2 <- tryCatch(sm$r.squared, error = function(e) NA_real_)
    lm_note <- sprintf(
      "Linear fit: R2 = %.3f; p = %s",
      ifelse(is.finite(fit_r2), fit_r2, NA_real_),
      ifelse(is.finite(fit_p), formatC(fit_p, format = "e", digits = 2), "NA")
    )
  }

  p_scatter_2 <- ggplot(df_plot, aes(x = nse_base, y = delta_nse, color = change_direction)) +
    geom_point(alpha = 0.70, size = 1.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    geom_smooth(method = "lm", se = FALSE, color = "#4d4d4d", linewidth = 0.7) +
    annotate("text", x = Inf, y = -Inf, label = gsub("; ", "\n", lm_note, fixed = TRUE), hjust = 1.02, vjust = -0.25, size = 4.2, family = "Times New Roman") +
    scale_color_manual(values = c(Improved = "#1b9e77", Degraded = "#d95f02"), name = "Change direction", guide = guide_legend(order = 2, nrow = 1)) +
    labs(
      title = "(e) NSE improvement relative to baseline performance",
      x = "Baseline model NSE",
      y = expression(Delta * "  NSE")
    ) +
    base_theme +
    theme(
      axis.title.x = element_text(size = 13, face = "plain"),
      axis.title.y = element_text(size = 13, face = "plain")
    )

  layout_design <- "
AA
AA
BC
DE
"

  p_map + p_box_base + p_box_mod + p_scatter_1 + p_scatter_2 +
    patchwork::plot_layout(design = layout_design, guides = "collect") &
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.box.just = "center",
      legend.key.width = grid::unit(1.0, "cm"),
      legend.key.height = grid::unit(0.46, "cm"),
      legend.spacing.x = grid::unit(0.14, "cm"),
      legend.spacing.y = grid::unit(0.12, "cm")
    )
}

main <- function() {
  args <- parse_args()

  rf_csv <- normalizePath(path.expand(args$rf_csv), winslash = "/", mustWork = FALSE)
  stations_csv <- normalizePath(path.expand(args$stations_csv), winslash = "/", mustWork = FALSE)
  sparse_root <- resolve_sparse_root(args$sparse_output_root)

  if (!file.exists(rf_csv)) {
    stop(sprintf("RF result CSV not found: %s", rf_csv))
  }
  if (!file.exists(stations_csv)) {
    stop(sprintf("Stations CSV not found: %s", stations_csv))
  }
  if (!dir.exists(sparse_root)) {
    stop(sprintf("Sparse Transformer output root not found: %s", sparse_root))
  }

  rf <- readr::read_csv(rf_csv, show_col_types = FALSE)
  stations <- readr::read_csv(stations_csv, show_col_types = FALSE)

  needed_rf <- c("station_id", "nse_base", "nse_mod", "mse_base", "mse_mod", "kge_base", "kge_mod")
  miss_rf <- setdiff(needed_rf, names(rf))
  if (length(miss_rf) > 0) {
    stop(sprintf("RF CSV missing required columns: %s", paste(miss_rf, collapse = ", ")))
  }

  needed_st <- c("gauge_id", "lat", "lon")
  miss_st <- setdiff(needed_st, names(stations))
  if (length(miss_st) > 0) {
    stop(sprintf("stations.csv missing required columns: %s", paste(miss_st, collapse = ", ")))
  }

  rf <- rf %>%
    mutate(
      station_id = pad_station_id(station_id),
      delta_nse = as.numeric(nse_mod) - as.numeric(nse_base)
    )

  stations <- stations %>%
    transmute(
      station_id = pad_station_id(gauge_id),
      lat = as.numeric(lat),
      lon = as.numeric(lon)
    )

  sparse_good <- load_sparse_good_stations(sparse_root, threshold = args$sparse_nse_threshold)
  if (length(sparse_good) == 0) {
    stop(sprintf(
      "No stations satisfy Sparse Transformer NSE > %.2f under %s",
      args$sparse_nse_threshold,
      sparse_root
    ))
  }

  df_plot <- rf %>%
    filter(station_id %in% sparse_good) %>%
    left_join(stations, by = "station_id") %>%
    filter(is.finite(lat), is.finite(lon))

  if (nrow(df_plot) < 5) {
    stop(sprintf(
      "Too few matched stations after filtering (n=%d). Check paths and threshold.",
      nrow(df_plot)
    ))
  }

  usa_map <- ggplot2::map_data("state")
  fig <- make_figure(df_plot, usa_map)

  out_figure <- normalizePath(path.expand(args$out_figure), winslash = "/", mustWork = FALSE)
  out_subset_csv <- normalizePath(path.expand(args$out_subset_csv), winslash = "/", mustWork = FALSE)
  dir.create(dirname(out_figure), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(out_subset_csv), recursive = TRUE, showWarnings = FALSE)

  ggplot2::ggsave(
    filename = out_figure,
    plot = fig,
    width = args$width,
    height = args$height,
    dpi = args$dpi,
    limitsize = FALSE
  )

  readr::write_csv(df_plot, out_subset_csv)

  message(sprintf("Sparse source: %s", sparse_root))
  message(sprintf("Threshold: Sparse Transformer NSE > %.2f", args$sparse_nse_threshold))
  message(sprintf("Matched stations: %d", nrow(df_plot)))
  message(sprintf("Figure saved: %s", out_figure))
  message(sprintf("Subset table saved: %s", out_subset_csv))
}

main()
