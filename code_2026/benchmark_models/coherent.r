##### Coherent Product-Ratio — Expanding Window #####

# The fitting process uses two rounds per cutoff:
# 1) Full-history countries (data back to 1959)
# 2) Restricted-years set for remaining countries (common years intersection)
#
# Note: The forecast() call sometimes prints
#   Error in stats::arima(x = …): non-stationary AR part from CSS
# This is expected — see original coherent.r for explanation.

library(demography)
library(tidyverse)
library(reshape2)
library(glue)
library(here)
library(jsonlite)

# load config
config <- fromJSON(file.path(here(), "code_2026", "benchmark_models", "config.json"))
cutoff_years <- config$cutoff_years
max_year     <- config$max_year

# load full country data (all years)
path <- here("data")
country_data <- read.table(paste(path, "country_training.txt", sep = "/"), header = FALSE,
                           col.names = c("Country", "Gender", "Year", "Age", "Rate"))
if (file.exists(paste(path, "country_test.txt", sep = "/"))) {
  country_test <- read.table(paste(path, "country_test.txt", sep = "/"), header = FALSE,
                             col.names = c("Country", "Gender", "Year", "Age", "Rate"))
  country_data <- rbind(country_data, country_test)
}
if (file.exists(paste(path, "country_final_test.txt", sep = "/"))) {
  country_final <- read.table(paste(path, "country_final_test.txt", sep = "/"), header = FALSE,
                              col.names = c("Country", "Gender", "Year", "Age", "Rate"))
  country_data <- rbind(country_data, country_final)
}
country_data <- country_data |> distinct() |> arrange(Country, Gender, Year, Age)

ages_all <- sort(unique(country_data$Age))

# helper: build list of demogdata objects
build_demog_list <- function(df, ages_vec) {
  data_objects <- list()
  for (c in unique(df$Country)) {
    for (g in unique(df$Gender)) {
      sub <- df %>% filter(Country == c, Gender == g)
      years_i <- sort(unique(sub$Year))

      mx_df <- sub %>%
        pivot_wider(names_from = "Year", values_from = "Rate") %>%
        select(-Country, -Gender, -Age)
      mx_mat <- as.matrix(mx_df)
      mx_mat[mx_mat == 0 | is.na(mx_mat)] <- 9e-06

      Ext <- matrix(1, nrow = nrow(mx_mat), ncol = ncol(mx_mat))

      data <- demogdata(
        data = mx_mat, pop = Ext, ages = ages_vec, years = years_i,
        type = "mortality", label = c, name = g
      )
      list_name <- glue("{c}_{g}")
      data_objects[[list_name]] <- data
    }
  }
  data_objects
}

# helper: collapse demogdata list into combined object
combine_demog_list <- function(dlist) {
  template <- dlist[[1]]
  template$rate <- lapply(dlist, function(x) x$rate[[1]])
  template$pop  <- lapply(dlist, function(x) x$pop[[1]])
  template
}

# Germany indices (causing fitting issues)
germany <- c(58, 59)

# --- Main loop over cutoff years ---
for (cutoff in cutoff_years) {
  cat("\n========== Coherent: cutoff =", cutoff, "==========\n")

  training <- country_data |> filter(Year <= cutoff)
  forecasted_years <- (cutoff + 1):max_year
  h <- length(forecasted_years)

  if (h <= 0) {
    cat("  No forecast horizon for cutoff", cutoff, ", skipping.\n")
    next
  }

  # Re-evaluate excluded countries per cutoff
  # (countries whose data doesn't start at min_year for this window)
  excluded_countries <- c()
  for (country in unique(training$Country)) {
    one_country <- training |> filter(Country == country)
    year_min <- min(one_country$Year, na.rm = TRUE)
    n_years <- length(unique(one_country$Year))
    if (year_min > 1959 | n_years < 10) {
      excluded_countries <- c(excluded_countries, country)
    }
  }
  cat("  Excluded countries:", length(excluded_countries), "\n")

  tryCatch({
    # Full-history set (excluded countries removed)
    full_hist_df <- training %>% filter(!Country %in% excluded_countries)

    if (length(unique(full_hist_df$Country)) < 2) {
      cat("  Not enough countries for coherent model, skipping cutoff", cutoff, "\n")
      next
    }

    full_demog_list <- build_demog_list(full_hist_df, ages_all)
    combined_full   <- combine_demog_list(full_demog_list)

    # Restricted-years set (all countries except Germany, common years)
    common_years <- training %>%
      group_by(Country, Gender) %>%
      summarise(yrs = list(unique(Year)), .groups = "drop") %>%
      pull(yrs) %>%
      reduce(intersect)

    df_restricted <- training %>%
      filter(Year %in% common_years, !Country %in% germany)

    if (length(unique(df_restricted$Country)) < 2) {
      cat("  Not enough countries for restricted coherent model, using full only.\n")
      demog_list_restricted <- NULL
    } else {
      demog_list_restricted <- build_demog_list(df_restricted, ages_all)
      combined_restricted   <- combine_demog_list(demog_list_restricted)
    }

    # Fit coherent models
    fit_full <- coherentfdm(combined_full)
    fc_full  <- forecast(fit_full, h = h)

    fc_restrict <- NULL
    fit_restrict <- NULL
    if (!is.null(demog_list_restricted)) {
      fit_restrict <- coherentfdm(combined_restricted)
      fc_restrict  <- forecast(fit_restrict, h = h)
    }

    # Build unified pop labels
    pop_labels <- names(full_demog_list)
    if (!is.null(demog_list_restricted)) {
      pop_labels <- union(pop_labels, names(demog_list_restricted))
    }

    # Extract forecasts
    fc_mixed <- map(pop_labels, function(label) {
      country_id <- as.integer(str_split(label, "_", simplify = TRUE)[1])
      if (country_id %in% excluded_countries && !is.null(fc_restrict)) {
        fc_obj <- fc_restrict[[label]]
      } else {
        fc_obj <- fc_full[[label]]
      }
      if (is.null(fc_obj)) return(NULL)
      list(
        rate  = fc_obj$rate[[1]],
        lower = fc_obj$rate$lower,
        upper = fc_obj$rate$upper
      )
    })
    names(fc_mixed) <- pop_labels
    fc_mixed <- compact(fc_mixed)

    # Extract fitted values
    fitted_mixed <- map(pop_labels, function(label) {
      country_id <- as.integer(str_split(label, "_", simplify = TRUE)[1])
      if (country_id %in% excluded_countries && !is.null(fit_restrict)) {
        fit_obj <- fit_restrict
      } else {
        fit_obj <- fit_full
      }
      tryCatch(
        exp(fit_obj$product$fitted$y + fit_obj$ratio[[label]]$fitted$y),
        error = function(e) NULL
      )
    })
    names(fitted_mixed) <- pop_labels
    fitted_mixed <- compact(fitted_mixed)

    # Convert fitted to long format
    fitted_results <- imap(fitted_mixed, function(mat, label) {
      df <- as.data.frame(mat)
      df$age <- ages_all
      long_df <- melt(df, id.vars = "age",
                      variable.name = "year", value.name = "rate")
      label_split <- str_split(label, "_", simplify = TRUE)
      country_id <- as.numeric(label_split[1])
      gender_id  <- as.numeric(label_split[2])
      if (country_id %in% excluded_countries) {
        years_used <- sort(unique(df_restricted$Year[df_restricted$Country == country_id]))
      } else {
        years_used <- sort(unique(full_hist_df$Year[full_hist_df$Country == country_id]))
      }
      long_df$year <- rep(years_used, each = length(ages_all))
      long_df$country <- country_id
      long_df$gender  <- gender_id
      long_df |> select(country, gender, year, age, rate)
    })

    # Convert forecasts to long format
    forecasted_results <- imap(fc_mixed, function(obj, label) {
      df <- as.data.frame(obj$rate)
      df$age <- ages_all
      long_df <- melt(df, id.vars = "age",
                      variable.name = "year", value.name = "rate")
      long_df$year <- rep(forecasted_years, each = length(ages_all))

      df_lower <- as.data.frame(obj$lower)
      df_lower$age <- ages_all
      lower_long <- melt(df_lower, id.vars = "age",
                         variable.name = "year", value.name = "lower_95")
      df_upper <- as.data.frame(obj$upper)
      df_upper$age <- ages_all
      upper_long <- melt(df_upper, id.vars = "age",
                         variable.name = "year", value.name = "upper_95")

      long_df$lower_95 <- lower_long$lower_95
      long_df$upper_95 <- upper_long$upper_95

      label_split <- str_split(label, "_", simplify = TRUE)
      long_df$country <- as.numeric(label_split[1])
      long_df$gender  <- as.numeric(label_split[2])
      long_df |> select(country, gender, year, age, rate, lower_95, upper_95)
    })

    # Assemble fitted results
    final_fitted_df <- bind_rows(fitted_results)
    final_fitted_df <- final_fitted_df |>
      transmute(geo = country, gender, year, age, mu = rate,
                total_var = NA_real_, aleatoric_var = NA_real_,
                epistemic_var = NA_real_, lower_95 = NA_real_,
                upper_95 = NA_real_)

    # Assemble forecast results
    final_forecasted_df <- bind_rows(forecasted_results)
    final_forecasted_df <- final_forecasted_df |>
      transmute(geo = country, gender, year, age, mu = rate,
                total_var = ((upper_95 - lower_95) / (2 * 1.96))^2,
                aleatoric_var = NA_real_, epistemic_var = NA_real_,
                lower_95, upper_95)

    # Save with cutoff in filename
    write.table(final_fitted_df,
                paste(path, glue("coherent_fitted_cutoff{cutoff}.txt"), sep = "/"),
                sep = " ", col.names = FALSE, row.names = FALSE)
    write.table(final_forecasted_df,
                paste(path, glue("coherent_forecast_cutoff{cutoff}.txt"), sep = "/"),
                sep = " ", col.names = FALSE, row.names = FALSE)

    cat("  Saved results for cutoff", cutoff, "\n")

  }, error = function(e) {
    cat("  FAILED for cutoff", cutoff, ":", conditionMessage(e), "\n")
  })
}
