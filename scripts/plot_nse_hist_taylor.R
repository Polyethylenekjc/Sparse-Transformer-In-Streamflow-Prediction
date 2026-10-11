#!/usr/bin/env Rscript

required_pkgs <- c("jsonlite", "dplyr", "ggplot2", "reticulate", "patchwork", "svglite")
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
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(reticulate)
  library(patchwork)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    output_root = "/mnt/d/Store/Sparse_Transformer",
    save_dir = "outputs/figures",
    bins = 25
  )

  if (length(args) == 0) {
    return(out)
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--output_root" && i + 1 <= length(args)) {
      out$output_root <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--save_dir" && i + 1 <= length(args)) {
      out$save_dir <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--bins" && i + 1 <= length(args)) {
      out$bins <- as.integer(args[[i + 1]])
      i <- i + 2
    } else {
      stop(sprintf("Unknown or incomplete arg: %s", key))
    }
  }

  out
}

get_station_dirs <- function(output_root) {
  dirs <- list.dirs(output_root, recursive = FALSE, full.names = TRUE)
  dirs[file.info(dirs)$isdir]
}

read_station_summary_nse <- function(station_dir) {
  summary_path <- file.path(station_dir, "summary.json")
  if (!file.exists(summary_path)) {
    return(NA_real_)
  }

  j <- tryCatch(fromJSON(summary_path), error = function(e) NULL)
  if (is.null(j) || is.null(j$metrics) || is.null(j$metrics$nse)) {
    return(NA_real_)
  }

  as.numeric(j$metrics$nse)
}

read_npz_arrays <- function(npz_path, np_module) {
  if (!file.exists(npz_path)) {
    return(NULL)
  }

  npz <- tryCatch(np_module$load(npz_path), error = function(e) NULL)
  if (is.null(npz)) {
    return(NULL)
  }

  keys <- tryCatch(py_to_r(npz$files), error = function(e) character())
  if (!("y" %in% keys) || !("p" %in% keys)) {
    try(npz$close(), silent = TRUE)
    return(NULL)
  }

  y <- tryCatch(as.numeric(py_to_r(npz[["y"]])), error = function(e) numeric())
  p <- tryCatch(as.numeric(py_to_r(npz[["p"]])), error = function(e) numeric())
  try(npz$close(), silent = TRUE)

  if (length(y) == 0 || length(p) == 0) {
    return(NULL)
  }

  list(y = y, p = p)
}

compute_station_metrics <- function(station_dir, np_module) {
  sid <- basename(station_dir)
  nse <- read_station_summary_nse(station_dir)

  npz_path <- file.path(station_dir, "test_arrays.npz")
  arr <- read_npz_arrays(npz_path, np_module)
  if (is.null(arr)) {
    return(data.frame(
      station_id = sid,
      nse = nse,
      corr = NA_real_,
      sd_obs = NA_real_,
      sd_pred = NA_real_,
      sd_ratio = NA_real_,
      crmsd = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  y <- arr$y
  p <- arr$p
  n <- min(length(y), length(p))
  if (n < 3) {
    return(data.frame(
      station_id = sid,
      nse = nse,
      corr = NA_real_,
      sd_obs = NA_real_,
      sd_pred = NA_real_,
      sd_ratio = NA_real_,
      crmsd = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  y <- y[seq_len(n)]
  p <- p[seq_len(n)]
  valid <- is.finite(y) & is.finite(p)
  y <- y[valid]
  p <- p[valid]

  if (length(y) < 3) {
    return(data.frame(
      station_id = sid,
      nse = nse,
      corr = NA_real_,
      sd_obs = NA_real_,
      sd_pred = NA_real_,
      sd_ratio = NA_real_,
      crmsd = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  sd_obs <- stats::sd(y)
  sd_pred <- stats::sd(p)
  corr <- suppressWarnings(stats::cor(y, p))

  if (!is.finite(nse)) {
    denom <- sum((y - mean(y))^2)
    if (denom > 0) {
      nse <- 1 - sum((p - y)^2) / denom
    }
  }

  sd_ratio <- if (is.finite(sd_obs) && sd_obs > 0) sd_pred / sd_obs else NA_real_
  crmsd <- if (is.finite(sd_ratio) && is.finite(corr)) {
    sqrt(max(0, 1 + sd_ratio^2 - 2 * sd_ratio * corr))
  } else {
    NA_real_
  }

  data.frame(
    station_id = sid,
    nse = as.numeric(nse),
    corr = as.numeric(corr),
    sd_obs = as.numeric(sd_obs),
    sd_pred = as.numeric(sd_pred),
    sd_ratio = as.numeric(sd_ratio),
    crmsd = as.numeric(crmsd),
    stringsAsFactors = FALSE
  )
}

build_nse_histogram <- function(df, bins, palette_vals) {
  nse_vals <- df$nse[is.finite(df$nse) & df$nse >= 0.5]
  if (length(nse_vals) < 1) {
    message("Skip histogram: no valid NSE values (>= 0.5).")
    return(NULL)
  }

  h <- hist(nse_vals, breaks = seq(0.5, 1.0, length.out = bins + 1), plot = FALSE)
  hist_df <- data.frame(
    xmin = h$breaks[-length(h$breaks)],
    xmax = h$breaks[-1],
    count = h$counts
  )
  hist_df$center <- (hist_df$xmin + hist_df$xmax) / 2

  p <- ggplot(hist_df) +
    geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = count, fill = center),
      color = "white",
      linewidth = 0.5
    ) +
    scale_fill_gradientn(
      colours = palette_vals,
      limits = c(0.5, 1.0),
      name = "NSE"
    ) +
    labs(
      x = "NSE",
      y = "Station Count"
    ) +
    theme_minimal(base_size = 14, base_family = "Times New Roman") +
    theme(
      legend.position = "none",
      panel.grid.major = element_line(color = "grey95", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.8),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      axis.title = element_text(size = 16, face = "plain", color = "black", family = "Times New Roman"),
      axis.text = element_text(size = 13, color = "black", family = "Times New Roman"),
      axis.ticks = element_line(color = "black", linewidth = 0.8),
      axis.ticks.length = unit(3.5, "pt")
    )

  return(p)
}

build_taylor_diagram <- function(df, palette_vals, nse_limits) {
  tdf <- df %>%
    filter(
      nse >= 0.5,
      is.finite(nse),
      is.finite(corr),
      is.finite(sd_ratio),
      corr >= 0,
      corr <= 1,
      sd_ratio >= 0
    )

  if (nrow(tdf) < 1) {
    message("Skip Taylor diagram: no valid station stats.")
    return(NULL)
  }

  tdf <- tdf %>%
    mutate(
      x = sd_ratio * corr,
      y = sd_ratio * sqrt(pmax(0, 1 - corr^2))
    )

  r_max <- max(c(1, tdf$sd_ratio), na.rm = TRUE)
  r_max <- max(1.1, r_max * 1.1)

  theta <- seq(0, pi / 2, length.out = 240)
  circle_levels <- pretty(c(0, r_max), n = 6)
  circle_levels <- circle_levels[circle_levels >= 0 & circle_levels <= r_max]

  circle_df <- do.call(rbind, lapply(circle_levels, function(r) {
    data.frame(x = r * cos(theta), y = r * sin(theta), level = r)
  }))

  corr_levels <- seq(0.1, 0.9, by = 0.1)
  ray_df <- do.call(rbind, lapply(corr_levels, function(cval) {
    data.frame(x = c(0, r_max * cval), y = c(0, r_max * sqrt(1 - cval^2)), corr = cval)
  }))

  r_seq <- seq(0, r_max, length.out = 180)
  th_seq <- seq(0, pi / 2, length.out = 220)
  grid_df <- expand.grid(r = r_seq, th = th_seq)
  grid_df$x <- grid_df$r * cos(grid_df$th)
  grid_df$y <- grid_df$r * sin(grid_df$th)
  grid_df$corr <- cos(grid_df$th)
  grid_df$crmsd <- sqrt(pmax(0, 1 + grid_df$r^2 - 2 * grid_df$r * grid_df$corr))

  crmsd_levels <- pretty(range(tdf$crmsd, na.rm = TRUE), n = 5)
  crmsd_levels <- crmsd_levels[crmsd_levels > 0]

  p <- ggplot() +
    geom_contour(
      data = grid_df,
      aes(x = x, y = y, z = crmsd),
      breaks = crmsd_levels,
      color = "grey82",
      linewidth = 0.4
    ) +
    geom_path(
      data = circle_df,
      aes(x = x, y = y, group = level),
      color = "grey75",
      linewidth = 0.45
    ) +
    geom_segment(
      data = ray_df,
      aes(x = x[1], y = y[1], xend = x[2], yend = y[2], group = corr),
      color = "grey80",
      linewidth = 0.35
    ) +
    geom_point(aes(x = 1, y = 0), shape = 16, size = 4.0, color = "red") + geom_text(aes(x = 1, y = 0.05, label = "Obs."), color = "red", family = "Times New Roman", size = 5) +
    geom_point(
      data = tdf,
      aes(x = x, y = y, color = nse),
      size = 3.2,
      alpha = 0.85,
      stroke = 0.6
    ) +
    scale_color_gradientn(colours = palette_vals, limits = c(0.5, 1.0), name = "NSE") +
    coord_equal(xlim = c(0, r_max), ylim = c(0, r_max), expand = FALSE) +
    labs(
      x = "Normalized SD × r",
      y = "Normalized SD × √(1−r²)"
    ) +
    theme_minimal(base_size = 14, base_family = "Times New Roman") +
    theme(
      panel.grid.major = element_line(color = "grey95", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.8),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = "none",
      axis.title = element_text(size = 16, face = "plain", color = "black", family = "Times New Roman"),
      axis.text = element_text(size = 13, color = "black", family = "Times New Roman"),
      axis.ticks = element_line(color = "black", linewidth = 0.8),
      axis.ticks.length = unit(3.5, "pt")
    )

  return(p)
}

main <- function() {
  cfg <- parse_args()

  if (!dir.exists(cfg$output_root)) {
    stop(sprintf("output_root does not exist: %s", cfg$output_root))
  }

  if (!dir.exists(cfg$save_dir)) {
    dir.create(cfg$save_dir, recursive = TRUE)
  }

  np <- import("numpy")

  station_dirs <- get_station_dirs(cfg$output_root)
  if (length(station_dirs) == 0) {
    stop(sprintf("No station directories found in: %s", cfg$output_root))
  }

  records <- lapply(station_dirs, function(d) compute_station_metrics(d, np))
  df <- bind_rows(records) %>% arrange(station_id)

  csv_file <- file.path(cfg$save_dir, "station_metrics_for_taylor.csv")
  write.csv(df, csv_file, row.names = FALSE)
  message(sprintf("Saved station metrics table: %s", csv_file))

  # Match the requested Streamlit-like NSE gradient endpoints.
  palette_vals <- c(
    "#440457", "#482878", "#3E4989", "#31688E", "#26828E",
    "#1F9E89", "#35B779", "#6CCE59", "#B4DE2C", "#FAE625"
  )

  nse_vals <- df$nse[is.finite(df$nse)]
  if (length(nse_vals) == 0) {
    stop("No valid NSE values found in summary.json files.")
  }
  nse_limits <- range(nse_vals, na.rm = TRUE)

  # Build individual plots
  hist_plot <- build_nse_histogram(df, cfg$bins, palette_vals)
  taylor_plot <- build_taylor_diagram(df, palette_vals, nse_limits)

  # Combine plots side-by-side (NSE left, Taylor right)
  if (!is.null(hist_plot) && !is.null(taylor_plot)) {
    combined_plot <- hist_plot | taylor_plot
    combined_out <- file.path(cfg$save_dir, "nse_taylor_combined.svg")
    ggsave(combined_out, plot = combined_plot, width = 16, height = 7.5, bg = "white")
    message(sprintf("Saved combined plot: %s", combined_out))
  } else {
    message("Could not generate combined plot (missing hist or taylor plot)")
  }

  message("Done.")
}

main()
