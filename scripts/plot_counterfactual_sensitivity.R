#!/usr/bin/env Rscript

required_pkgs <- c("ggplot2", "dplyr", "patchwork")
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
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    station_csv = "outputs/sensitive/physical_sensitivity_station_table.csv",
    out_dir = "outputs/sensitive/figures",
    prefix = "Fig_counterfactual"
  )

  if (length(args) == 0) {
    return(out)
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--station_csv" && i + 1 <= length(args)) {
      out$station_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out_dir" && i + 1 <= length(args)) {
      out$out_dir <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--prefix" && i + 1 <= length(args)) {
      out$prefix <- args[[i + 1]]
      i <- i + 2
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }

  out
}

pick_font_family <- function() {
  target <- "Times New Roman"
  if (requireNamespace("systemfonts", quietly = TRUE)) {
    fam <- tryCatch(systemfonts::system_fonts()$family, error = function(e) character())
    if (any(tolower(fam) == tolower(target))) {
      return(target)
    }
  }
  return("serif")
}

panel_title <- function(kind) {
  mapping <- c(
    rain_scale = "(b) Rain Scaling",
    rain_pattern = "(c) Rainfall Pattern",
    streamflow_scaling = "(d) Streamflow Scaling",
    temp_plus = "(e) Temperature Increase"
  )
  ifelse(kind %in% names(mapping), mapping[kind], kind)
}

kind_label <- function(kind) {
  mapping <- c(
    rain_scale = "Rain Scale",
    streamflow_scaling = "Streamflow Scaling",
    temp_plus = "Temperature Increase",
    rain_pattern = "Rainfall Pattern"
  )
  ifelse(kind %in% names(mapping), mapping[kind], kind)
}

label_setting <- function(kind, value) {
  kind <- as.character(kind)
  value <- as.character(value)

  if (kind == "rain_scale") {
    return(sprintf("x%s", value))
  }
  if (kind == "streamflow_scaling") {
    v <- suppressWarnings(as.numeric(value))
    if (is.na(v)) return(value)
    return(sprintf("x%.1f", v))
  }
  if (kind == "temp_plus") {
    v <- suppressWarnings(as.numeric(value))
    if (is.na(v)) return(value)
    return(sprintf("+%.1f C", v))
  }
  if (kind == "rain_pattern") {
    txt <- gsub("_", " ", value)
    return(tools::toTitleCase(txt))
  }
  value
}

prepare_data <- function(df) {
  req <- c(
    "scenario_kind", "scenario_value", "delta_mse", "base_mse",
    "delta_peak_time", "delta_volume", "base_volume"
  )
  miss <- setdiff(req, colnames(df))
  if (length(miss) > 0) {
    stop(sprintf("Missing columns: %s", paste(miss, collapse = ", ")))
  }

  out <- df %>%
    mutate(
      scenario_kind = ifelse(as.character(scenario_kind) == "runoff_scaling", "streamflow_scaling", as.character(scenario_kind)),
      scenario_value = as.character(scenario_value),
      panel = panel_title(scenario_kind),
      setting = mapply(label_setting, scenario_kind, scenario_value, USE.NAMES = FALSE),
      scenario_kind_label = kind_label(scenario_kind),
      delta_mse_pct = 100 * delta_mse / pmax(abs(base_mse), 1e-9),
      delta_volume_pct = 100 * delta_volume / pmax(abs(base_volume), 1e-9),
      abs_peak_shift = abs(delta_peak_time),
      response_amplitude = sqrt(delta_mse_pct^2 + delta_volume_pct^2)
    )

  if (!("expected_behavior" %in% colnames(out))) {
    out$expected_behavior <- out$delta_mse > 0
  }
  out$expected_behavior <- as.logical(out$expected_behavior)

  if (!("nse_group" %in% colnames(out))) {
    if ("station_nse" %in% colnames(out)) {
      out$nse_group <- ifelse(is.finite(out$station_nse) & out$station_nse < 0.5, "NSE<0.5", "NSE>0.5")
    } else {
      out$nse_group <- "NSE>0.5"
    }
  }
  out$nse_group <- ifelse(out$nse_group %in% c("NSE<0.5", "NSE>0.5"), out$nse_group, "NSE>0.5")

  kind_order <- c("rain_scale", "rain_pattern", "streamflow_scaling", "temp_plus")
  panel_order <- c(
    "(b) Rain Scaling",
    "(c) Rainfall Pattern",
    "(d) Streamflow Scaling",
    "(e) Temperature Increase"
  )

  out$scenario_kind <- factor(out$scenario_kind, levels = kind_order)
  out$panel <- factor(out$panel, levels = panel_order)
  out$scenario_kind_label <- factor(
    out$scenario_kind_label,
    levels = c("Rain Scale", "Rainfall Pattern", "Streamflow Scaling", "Temperature Increase")
  )

  out <- out %>%
    filter(
      is.finite(delta_mse_pct),
      is.finite(delta_volume_pct),
      is.finite(delta_peak_time),
      is.finite(response_amplitude)
    ) %>%
    group_by(panel) %>%
    mutate(setting = factor(setting, levels = unique(setting))) %>%
    ungroup()

  out
}

plot_expected_compliance_top <- function(df, family_use, show_legend = TRUE) {
  df_plot <- df %>% filter(nse_group == "NSE>0.5")
  if (nrow(df_plot) < 1) {
    df_plot <- df
  }

  station_vote <- df_plot %>%
    mutate(
      scenario_kind_label = factor(
        scenario_kind_label,
        levels = c("Rain Scale", "Rainfall Pattern", "Streamflow Scaling", "Temperature Increase")
      )
    ) %>%
    group_by(station_id, scenario_kind_label) %>%
    summarise(
      expected_ratio = mean(as.logical(expected_behavior), na.rm = TRUE),
      station_expected = expected_ratio > 0.5,
      .groups = "drop"
    )

  plot_df <- station_vote %>%
    mutate(expected_label = ifelse(station_expected, "Expected", "Unexpected")) %>%
    group_by(scenario_kind_label, expected_label) %>%
    summarise(n = n(), .groups = "drop") %>%
    ungroup()

  plot_df$expected_label <- factor(plot_df$expected_label, levels = c("Expected", "Unexpected"))

  donut_df <- plot_df %>%
    group_by(scenario_kind_label) %>%
    mutate(
      total_n = sum(n, na.rm = TRUE),
      frac = n / pmax(total_n, 1),
      pct_label = paste0(round(100 * frac), "%")
    ) %>%
    ungroup()

  donut_df <- donut_df %>%
    mutate(
      fill_color = case_when(
        scenario_kind_label == "Rain Scale" & expected_label == "Expected" ~ "#2F80ED",
        scenario_kind_label == "Rain Scale" & expected_label == "Unexpected" ~ "#F2994A",
        scenario_kind_label == "Rainfall Pattern" & expected_label == "Expected" ~ "#8E44AD",
        scenario_kind_label == "Rainfall Pattern" & expected_label == "Unexpected" ~ "#F2C94C",
        scenario_kind_label == "Streamflow Scaling" & expected_label == "Expected" ~ "#27AE60",
        scenario_kind_label == "Streamflow Scaling" & expected_label == "Unexpected" ~ "#BB6BD9",
        scenario_kind_label == "Temperature Increase" & expected_label == "Expected" ~ "#EB5757",
        scenario_kind_label == "Temperature Increase" & expected_label == "Unexpected" ~ "#2D9CDB",
        TRUE ~ "#888888"
      )
    )

  sig_df <- station_vote %>%
    group_by(scenario_kind_label) %>%
    summarise(
      n_total = n(),
      n_expected = sum(station_expected, na.rm = TRUE),
      p_value = ifelse(n_total > 0, binom.test(n_expected, n_total, p = 0.5, alternative = "greater")$p.value, NA_real_),
      .groups = "drop"
    ) %>%
    mutate(
      sig_label = case_when(
        !is.finite(p_value) ~ "",
        p_value < 1e-3 ~ paste0("p=", formatC(p_value, format = "e", digits = 2)),
        TRUE ~ paste0("p=", formatC(p_value, format = "f", digits = 3))
      )
    )

  ggplot(donut_df, aes(x = 2, y = frac, fill = fill_color)) +
    geom_col(width = 0.52, color = "white", linewidth = 0.55, alpha = 0.95) +
    geom_text(
      aes(label = pct_label),
      position = position_stack(vjust = 0.5),
      size = 4.9,
      family = family_use,
      color = "white",
      fontface = "bold"
    ) +
    geom_text(
      data = sig_df,
      aes(x = 1.1, y = 0, label = sig_label),
      inherit.aes = FALSE,
      size = 5.3,
      family = family_use,
      fontface = "bold",
      color = "black"
    ) +
    coord_polar(theta = "y") +
    xlim(0.55, 2.45) +
    facet_wrap(~scenario_kind_label, ncol = 4) +
    scale_fill_identity() +
    labs(
      title = "(a) Station-Level Expected vs Unexpected (NSE > 0.5)",
      x = NULL,
      fill = NULL
    ) +
    theme_void(base_size = 17, base_family = family_use) +
    theme(
      plot.title = element_text(size = 20, face = "bold", hjust = 0.5, family = family_use, margin = margin(t = -4, b = 8)),
      strip.text = element_text(size = 18.5, face = "bold", family = family_use),
      legend.position = ifelse(show_legend, "bottom", "none"),
      legend.direction = "horizontal",
      legend.text = element_text(size = 16, family = family_use),
      plot.margin = margin(t = 8, r = 8, b = 2, l = 8)
    )
}

    plot_tradeoff_with_overlay <- function(df, family_use, show_legend = TRUE) {
  df_plot <- df %>% filter(nse_group == "NSE>0.5")
  if (nrow(df_plot) < 1) {
    df_plot <- df
  }

      df_plot <- df_plot %>%
        filter(
          is.finite(delta_mse_pct),
          is.finite(delta_volume_pct),
          !is.na(panel),
          !is.na(setting)
        )
      if (nrow(df_plot) < 1) {
        return(ggplot() + theme_void())
      }

  color_breaks <- c("Rain Scale", "Rainfall Pattern", "Streamflow Scaling", "Temperature Increase")

  panel_levels <- as.character(na.omit(unique(df_plot$panel)))
  panel_levels <- panel_levels[order(match(panel_levels, levels(df_plot$panel)))]

  build_one_panel <- function(panel_name) {
    d <- df_plot %>% filter(as.character(panel) == panel_name)
    if (nrow(d) < 1) {
      return(NULL)
    }

    x_lim <- quantile(abs(d$delta_volume_pct), probs = 0.9, na.rm = TRUE)
    y_lim <- quantile(abs(d$delta_mse_pct), probs = 0.9, na.rm = TRUE)
    if (!is.finite(x_lim) || x_lim <= 0) x_lim <- max(abs(d$delta_volume_pct), na.rm = TRUE)
    if (!is.finite(y_lim) || y_lim <= 0) y_lim <- max(abs(d$delta_mse_pct), na.rm = TRUE)
    if (!is.finite(x_lim) || x_lim <= 0) x_lim <- 1
    if (!is.finite(y_lim) || y_lim <= 0) y_lim <- 1

    line_base <- d %>%
      group_by(setting, scenario_kind, scenario_kind_label) %>%
      summarise(
        delta_mse_pct = median(delta_mse_pct, na.rm = TRUE),
        delta_volume_pct = median(delta_volume_pct, na.rm = TRUE),
        n_obs = n(),
        .groups = "drop"
      ) %>%
      arrange(scenario_kind, as.character(setting)) %>%
      mutate(x_rank = row_number())

    n_rank <- nrow(line_base)
    if (n_rank < 1) {
      line_base <- data.frame(
        setting = "",
        scenario_kind = NA,
        scenario_kind_label = NA,
        delta_mse_pct = 0,
        delta_volume_pct = 0,
        x_rank = 1
      )
      n_rank <- 1
    }

    if (n_rank == 1) {
      line_base$x_line <- 0
    } else {
      line_base$x_line <- -x_lim + (line_base$x_rank - 1) * (2 * x_lim / (n_rank - 1))
    }

    line_overlay <- bind_rows(
      line_base %>% transmute(metric = "Relative Error Change (%)", setting, x_line, value = delta_mse_pct),
      line_base %>% transmute(metric = "Relative Streamflow Change (%)", setting, x_line, value = delta_volume_pct)
    )

    line_max <- max(abs(line_overlay$value), na.rm = TRUE)
    if (!is.finite(line_max) || line_max <= 0) line_max <- 1
    line_scale <- y_lim / (2.4 * line_max)
    y_shift <- -0.55 * y_lim
    line_overlay <- line_overlay %>%
      mutate(y_line = value * line_scale + y_shift)

    line_band <- data.frame(
      xmin = -x_lim,
      xmax = x_lim,
      ymin = -y_lim,
      ymax = y_lim
    )

    qbg <- data.frame(
      xmin = c(-x_lim, 0, -x_lim, 0),
      xmax = c(0, x_lim, 0, x_lim),
      ymin = c(0, 0, -y_lim, -y_lim),
      ymax = c(y_lim, y_lim, 0, 0),
      quad = c("Q2", "Q1", "Q4", "Q3")
    )

    line_breaks <- unique(line_base$x_line)
    line_labels <- as.character(line_base$setting)

    ggplot(d, aes(x = delta_volume_pct, y = delta_mse_pct, color = scenario_kind_label)) +
      geom_rect(
        data = qbg,
        aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = quad),
        inherit.aes = FALSE,
        alpha = 0.30,
        color = NA
      ) +
      geom_hline(yintercept = 0, color = "#BDBDBD", linewidth = 0.45) +
      geom_vline(xintercept = 0, color = "#BDBDBD", linewidth = 0.45) +
      geom_point(size = 2.0, alpha = 0.62) +
      geom_rect(
        data = line_band,
        aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
        inherit.aes = FALSE,
        fill = "white",
        alpha = 0.36,
        color = NA
      ) +
      geom_line(
        data = line_overlay %>% filter(metric == "Relative Error Change (%)"),
        aes(x = x_line, y = y_line),
        inherit.aes = FALSE,
        color = "#D35400",
        linewidth = 0.85,
        alpha = 0.95
      ) +
      geom_point(
        data = line_overlay %>% filter(metric == "Relative Error Change (%)"),
        aes(x = x_line, y = y_line),
        inherit.aes = FALSE,
        color = "#D35400",
        size = 1.7,
        alpha = 0.95
      ) +
      geom_line(
        data = line_overlay %>% filter(metric == "Relative Streamflow Change (%)"),
        aes(x = x_line, y = y_line),
        inherit.aes = FALSE,
        color = "#1F78B4",
        linewidth = 0.85,
        alpha = 0.95
      ) +
      geom_point(
        data = line_overlay %>% filter(metric == "Relative Streamflow Change (%)"),
        aes(x = x_line, y = y_line),
        inherit.aes = FALSE,
        color = "#1F78B4",
        size = 1.7,
        alpha = 0.95
      ) +
      scale_fill_manual(values = c(
        Q1 = "#FFB8B8",
        Q2 = "#FFD6A3",
        Q3 = "#BFDFFF",
        Q4 = "#CBEBCB"
      ), guide = "none") +
      scale_color_manual(values = c(
        "Rain Scale" = "#2F80ED",
        "Streamflow Scaling" = "#27AE60",
        "Temperature Increase" = "#EB5757",
        "Rainfall Pattern" = "#8E44AD"
      ), breaks = color_breaks, drop = FALSE) +
      scale_x_continuous(
        limits = c(-x_lim, x_lim),
        position = "top",
        expand = c(0, 0),
        name = "Trade Axis: Relative Streamflow Change (%)",
        sec.axis = sec_axis(
          ~ .,
          breaks = line_breaks,
          labels = line_labels,
          name = "Line Axis: Scenario Setting"
        )
      ) +
      scale_y_continuous(
        limits = c(-y_lim, y_lim),
        expand = c(0, 0),
        name = "Trade Axis: Relative Error Change (%)",
        sec.axis = sec_axis(
          ~ ((. - y_shift) / line_scale),
          name = "Line Axis: Mean Relative Response (%)"
        )
      ) +
      labs(
        title = panel_name,
        x = NULL,
        y = NULL,
        color = "Scenario Type"
      ) +
      guides(color = guide_legend(order = 1, nrow = 1, byrow = TRUE)) +
      theme_minimal(base_size = 16, base_family = family_use) +
      theme(
        plot.title = element_text(size = 17.5, face = "bold", hjust = 0.5, family = family_use),
        axis.title.x.top = element_text(size = 12.8, family = family_use, color = "#8E44AD"),
        axis.text.x.top = element_text(size = 10.8, family = family_use, color = "#8E44AD"),
        axis.title.y.left = element_text(size = 12.8, family = family_use, color = "#8E44AD"),
        axis.text.y.left = element_text(size = 10.8, family = family_use, color = "#8E44AD"),
        axis.title.x.bottom = element_text(size = 12.2, family = family_use, color = "#1F78B4"),
        axis.text.x.bottom = element_text(size = 10.5, angle = 18, hjust = 1, family = family_use, color = "#1F78B4"),
        axis.title.y.right = element_text(size = 12.2, family = family_use, color = "#D35400"),
        axis.text.y.right = element_text(size = 10.5, family = family_use, color = "#D35400"),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.title = element_text(size = 15.5, family = family_use),
        legend.text = element_text(size = 15, family = family_use),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.45)
      ) +
      theme(legend.position = ifelse(show_legend, "bottom", "none"))
  }

  plots <- lapply(panel_levels, build_one_panel)
  plots <- Filter(Negate(is.null), plots)
  if (length(plots) < 1) {
    return(ggplot() + theme_void())
  }

  wrap_plots(plots, ncol = 2) +
    plot_annotation(
      title = "(b) Overlayed Scenario Map and Relative Response",
      theme = theme(
        plot.title = element_text(size = 17, face = "bold", hjust = 0.5, family = family_use)
      )
    )
}

plot_shared_legend <- function(df, family_use) {
  color_breaks <- c("Rain Scale", "Rainfall Pattern", "Streamflow Scaling", "Temperature Increase")

  d_color <- data.frame(
    x = seq_along(color_breaks),
    y = 3,
    scenario_kind_label = factor(color_breaks, levels = color_breaks)
  )
  d_line <- data.frame(
    x = c(1, 1),
    xend = c(2, 2),
    y = c(1, 1.5),
    yend = c(1, 1.5),
    metric = factor(
      c("Relative Error Change (%)", "Relative Streamflow Change (%)"),
      levels = c("Relative Error Change (%)", "Relative Streamflow Change (%)")
    )
  )

  ggplot() +
    geom_segment(data = d_line, aes(x = x, xend = xend, y = y, yend = yend, linetype = metric),
      linewidth = 1.0, color = "#444444", alpha = 0) +
    geom_point(data = d_color, aes(x = x, y = y, color = scenario_kind_label), alpha = 0, size = 3) +
    scale_color_manual(values = c(
      "Rain Scale" = "#2F80ED",
      "Streamflow Scaling" = "#27AE60",
      "Temperature Increase" = "#EB5757",
      "Rainfall Pattern" = "#8E44AD"
    ), breaks = color_breaks, drop = FALSE, name = NULL) +
    scale_linetype_manual(
      values = c(
        "Relative Error Change (%)" = "solid",
        "Relative Streamflow Change (%)" = "solid"
      ),
      name = "Line Meaning"
    ) +
    guides(
      color = guide_legend(order = 1, nrow = 1, byrow = TRUE, override.aes = list(alpha = 1, size = 3)),
      linetype = guide_legend(
        order = 2,
        nrow = 1,
        byrow = TRUE,
        override.aes = list(alpha = 1, color = c("#D35400", "#1F78B4"))
      )
    ) +
    theme_void(base_family = family_use) +
    theme(
      legend.position = "top",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.title = element_text(size = 15.5, family = family_use),
      legend.text = element_text(size = 15, family = family_use),
      plot.margin = margin(t = -8, r = 0, b = 0, l = 0)
    )
}

plot_peakshift_focus <- function(df, family_use) {
  peak_df <- df %>%
    filter(nse_group == "NSE>0.5") %>%
    filter(scenario_kind %in% c("streamflow_scaling", "runoff_scaling", "rain_pattern")) %>%
    mutate(shift_dir = ifelse(delta_peak_time < 0, "Earlier", ifelse(delta_peak_time > 0, "Later", "No Shift")))

  if (nrow(peak_df) < 1) {
    peak_df <- df %>%
      filter(scenario_kind %in% c("streamflow_scaling", "runoff_scaling", "rain_pattern")) %>%
      mutate(shift_dir = ifelse(delta_peak_time < 0, "Earlier", ifelse(delta_peak_time > 0, "Later", "No Shift")))
  }

  ggplot(peak_df, aes(x = setting, y = delta_peak_time, color = shift_dir)) +
    geom_hline(yintercept = 0, color = "#8C8C8C", linewidth = 0.45, linetype = "dashed") +
    geom_segment(aes(xend = setting, y = 0, yend = delta_peak_time), linewidth = 0.8, alpha = 0.9) +
    geom_point(size = 2.8) +
    facet_wrap(~panel, ncol = 2, scales = "free_x") +
    scale_color_manual(values = c(
      "Earlier" = "#1F78B4",
      "Later" = "#D62728",
      "No Shift" = "#6E6E6E"
    )) +
    guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
    labs(
      title = "(c) Peak-Time Shift under Timing-Related Counterfactuals",
      x = "Scenario Setting",
      y = "Peak-Time Shift (index)",
      color = "Direction"
    ) +
    theme_minimal(base_size = 16, base_family = family_use) +
    theme(
      plot.title = element_text(size = 19, face = "bold", hjust = 0.5, family = family_use),
      strip.text = element_text(size = 17.5, face = "bold", family = family_use),
      axis.title = element_text(size = 14, family = family_use),
      axis.text.x = element_text(size = 11.5, angle = 20, hjust = 1, family = family_use),
      axis.text.y = element_text(size = 11.5, family = family_use),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_text(size = 15.5, family = family_use),
      legend.text = element_text(size = 15, family = family_use),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.45)
    )
}

save_plot_pair <- function(plot_obj, png_path, pdf_path, w = 14, h = 9) {
  ggsave(
    png_path,
    plot_obj,
    width = w,
    height = h,
    dpi = 320,
    bg = "white",
    device = png,
    type = "cairo"
  )
  ggsave(
    pdf_path,
    plot_obj,
    width = w,
    height = h,
    bg = "white",
    device = cairo_pdf
  )
}

main <- function() {
  opt <- parse_args()

  if (!file.exists(opt$station_csv)) {
    stop(sprintf("station csv not found: %s", opt$station_csv))
  }

  dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)
  df <- read.csv(opt$station_csv, stringsAsFactors = FALSE)
  df <- prepare_data(df)

  family_use <- pick_font_family()

  p_top <- plot_expected_compliance_top(df, family_use, show_legend = FALSE)
  p_trade_overlay <- plot_tradeoff_with_overlay(df, family_use, show_legend = FALSE)
  p_peak <- plot_peakshift_focus(df, family_use)
  p_legend <- plot_shared_legend(df, family_use)

  p_main <- p_top / p_trade_overlay / p_legend +
    plot_layout(heights = c(1.7, 6.6, 0.9))

  f1_png <- file.path(opt$out_dir, sprintf("%s_panel_main.png", opt$prefix))
  f1_pdf <- file.path(opt$out_dir, sprintf("%s_panel_main.pdf", opt$prefix))
  f2_png <- file.path(opt$out_dir, sprintf("%s_panel_tradeoff.png", opt$prefix))
  f2_pdf <- file.path(opt$out_dir, sprintf("%s_panel_tradeoff.pdf", opt$prefix))
  f3_png <- file.path(opt$out_dir, sprintf("%s_panel_peakshift.png", opt$prefix))
  f3_pdf <- file.path(opt$out_dir, sprintf("%s_panel_peakshift.pdf", opt$prefix))

  save_plot_pair(p_main, f1_png, f1_pdf, w = 16, h = 18)
  save_plot_pair(p_trade_overlay, f2_png, f2_pdf, w = 14, h = 9)
  save_plot_pair(p_peak, f3_png, f3_pdf, w = 14, h = 9)

  message(sprintf("Font family used: %s", family_use))
  message("Saved:")
  message(f1_png)
  message(f1_pdf)
  message(f2_png)
  message(f2_pdf)
  message(f3_png)
  message(f3_pdf)
}

main()
