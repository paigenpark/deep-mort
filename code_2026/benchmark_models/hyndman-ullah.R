##### Hyndman-Ullah — Expanding Window #####

# vector of required packages
required_packages <- c("demography", "tidyverse", "reshape2", "glue", "here", "jsonlite")

# function to check and install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# apply the function to all required packages
invisible(lapply(required_packages, install_if_missing))

# load the libraries
lapply(required_packages, library, character.only = TRUE)

# load config
config <- fromJSON(file.path(here(), "code_2026", "benchmark_models", "config.json"))
cutoff_years <- config$cutoff_years
max_year     <- config$max_year

# load full country data (all years)
path <- here("data")
country_data <- read.table(paste(path, "country_training.txt", sep = "/"), header = FALSE)
if (file.exists(paste(path, "country_test.txt", sep = "/"))) {
  country_test <- read.table(paste(path, "country_test.txt", sep = "/"), header = FALSE)
  country_data <- rbind(country_data, country_test)
}
if (file.exists(paste(path, "country_final_test.txt", sep = "/"))) {
  country_final <- read.table(paste(path, "country_final_test.txt", sep = "/"), header = FALSE)
  country_data <- rbind(country_data, country_final)
}
colnames(country_data) <- c('Country', 'Gender', 'Year', 'Age', 'Rate')
country_data <- country_data |> distinct() |> arrange(Country, Gender, Year, Age)

ages <- sort(unique(country_data$Age))

# --- Main loop over cutoff years ---
for (cutoff in cutoff_years) {
  cat("\n========== Hyndman-Ullah: cutoff =", cutoff, "==========\n")

  training <- country_data |> filter(Year <= cutoff)
  countries <- unique(training$Country)
  genders <- unique(training$Gender)
  forecasted_years <- (cutoff + 1):max_year
  h <- length(forecasted_years)

  if (h <= 0) {
    cat("  No forecast horizon for cutoff", cutoff, ", skipping.\n")
    next
  }

  fitted_results <- list()
  forecasted_results <- list()

  for (i in countries) {
    for (j in genders) {
      tryCatch({
        filtered <- training |>
          filter(Country == i & Gender == j)
        years_i <- sort(unique(filtered$Year))

        if (length(years_i) < 10) {
          cat("  Skipping country", i, "gender", j, "— only", length(years_i), "years\n")
          next
        }

        mx_df <- filtered |>
          pivot_wider(names_from = 'Year', values_from = 'Rate') |>
          select(-Age, -Gender, -Country)
        mx_mat <- as.matrix(mx_df)
        mx_mat[mx_mat == 0 | is.na(mx_mat)] <- 9e-06

        Ext <- matrix(1, nrow = nrow(mx_mat), ncol = ncol(mx_mat))

        data <- demogdata(
          data = mx_mat, pop = Ext, ages = ages, years = years_i,
          type = "mortality", label = i, name = j
        )

        hu_output <- fdm(data, series = j, ages = ages, years = years_i, order = 3)

        # fitted results
        fitted <- exp(hu_output$fitted$y)
        df_fitted <- as.data.frame(fitted)
        df_fitted$age <- ages
        df_fitted_long <- melt(df_fitted, id.vars = "age",
                               variable.name = "year", value.name = "rate")
        df_fitted_long$year <- rep(years_i, each = length(ages))
        df_fitted_long$country <- i
        df_fitted_long$gender <- j
        fitted_results[[paste(i, j, sep = "_")]] <- df_fitted_long

        # forecasts with prediction intervals
        forecasted <- forecast(hu_output, h = h)
        forecasted_rates <- do.call(cbind, forecasted$rate[1])
        df_forecasted <- as.data.frame(forecasted_rates)
        df_forecasted$age <- ages
        df_forecasted_long <- melt(df_forecasted, id.vars = "age",
                                   variable.name = "year", value.name = "rate")
        df_forecasted_long$year <- rep(forecasted_years, each = length(ages))
        df_forecasted_long$country <- i
        df_forecasted_long$gender <- j

        lower_rates <- do.call(cbind, forecasted$rate$lower)
        upper_rates <- do.call(cbind, forecasted$rate$upper)
        df_lower <- as.data.frame(lower_rates)
        df_lower$age <- ages
        lower_long <- melt(df_lower, id.vars = "age",
                           variable.name = "year", value.name = "lower_95")
        df_upper <- as.data.frame(upper_rates)
        df_upper$age <- ages
        upper_long <- melt(df_upper, id.vars = "age",
                           variable.name = "year", value.name = "upper_95")
        df_forecasted_long$lower_95 <- lower_long$lower_95
        df_forecasted_long$upper_95 <- upper_long$upper_95

        forecasted_results[[paste(i, j, sep = "_")]] <- df_forecasted_long
      }, error = function(e) {
        cat("  Error for country", i, "gender", j, ":", conditionMessage(e), "\n")
      })
    }
  }

  # assemble fitted results
  final_fitted_df <- bind_rows(fitted_results)
  final_fitted_df <- final_fitted_df |>
    transmute(geo = country, gender, year, age, mu = rate,
              total_var = NA_real_, aleatoric_var = NA_real_,
              epistemic_var = NA_real_, lower_95 = NA_real_,
              upper_95 = NA_real_)

  # assemble forecast results
  final_forecasted_df <- bind_rows(forecasted_results)
  final_forecasted_df <- final_forecasted_df |>
    transmute(geo = country, gender, year, age, mu = rate,
              total_var = ((upper_95 - lower_95) / (2 * 1.96))^2,
              aleatoric_var = NA_real_, epistemic_var = NA_real_,
              lower_95, upper_95)

  # save with cutoff in filename
  write.table(final_fitted_df,
              paste(path, glue("hu_fitted_cutoff{cutoff}.csv"), sep = "/"),
              sep = ",", col.names = FALSE, row.names = FALSE)
  write.table(final_forecasted_df,
              paste(path, glue("hu_forecast_cutoff{cutoff}.csv"), sep = "/"),
              sep = ",", col.names = FALSE, row.names = FALSE)

  cat("  Saved results for cutoff", cutoff, "\n")
}
