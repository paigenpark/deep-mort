##### Hyndman-Ullah #####

# vector of required packages
required_packages <- c("demography", "tidyverse", "glue", "here")

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


path <- here("data")
country_training <- read.table(paste(path, "country_training.txt", sep = "/"), header = FALSE)

countries <- unique(country_training[,1])
genders <- unique(country_training[,2])
years <- unique(country_training[,3])
ages <- unique(country_training[,4])
forecasted_years <- 2006:2015
colnames(country_training) <- c('Country', 'Gender', 'Year', 'Age', 'Rate')

fitted_results <- list()
forecasted_results <- list()

for (i in countries) {
  for (j in genders) {
    filtered <- country_training |>
      filter(country_training[,1] == i & country_training[,2] == j)

    # get number of years available for country/gender combo
    years_i <- sort(unique(filtered$Year))

    mx_df <- filtered |>
      pivot_wider(names_from = 'Year',
                  values_from = 'Rate') |>
      select(-Age, -Gender, -Country)
    mx_mat <- as.matrix(mx_df)
    mx_mat[mx_mat == 0 | is.na(mx_mat)] <- 9e-06


    Ext <- matrix(1, nrow = nrow(mx_mat), ncol = ncol(mx_mat))

    data <- demogdata(
      data = mx_mat,
      pop = Ext,
      ages = ages,
      years = years_i,
      type = "mortality",
      label = i,
      name = j
    )

    hu_output <- fdm(data,
                     series = j,
                     ages = ages,
                     years = years_i,
                     order = 3)

    fitted <- exp(hu_output$fitted$y)
    df_fitted <- as.data.frame(fitted)
    colnames(df_fitted) <- years_i
    df_fitted$age <- ages
    df_fitted_long <- df_fitted |>
      pivot_longer(cols = -age, names_to = "year", values_to = "rate") |>
      mutate(year = as.integer(year))
    df_fitted_long$country <- i
    df_fitted_long$gender <- j
    fitted_results[[paste(i, j, sep = "_")]] <- df_fitted_long

    # get/prep forecasts (point estimate + 95% prediction intervals)
    forecasted <- forecast(hu_output, h=10, level=95)
    forecasted_rates <- do.call(cbind, forecasted$rate[1])
    df_forecasted <- as.data.frame(forecasted_rates)
    colnames(df_forecasted) <- forecasted_years
    df_forecasted$age <- ages
    df_forecasted_long <- df_forecasted |>
      pivot_longer(cols = -age, names_to = "year", values_to = "rate") |>
      mutate(year = as.integer(year))
    df_forecasted_long$country <- i
    df_forecasted_long$gender <- j

    # extract 95% prediction intervals
    lower_rates <- forecasted$rate$lower
    upper_rates <- forecasted$rate$upper
    df_lower <- as.data.frame(lower_rates)
    colnames(df_lower) <- forecasted_years
    df_lower$age <- ages
    lower_long <- df_lower |>
      pivot_longer(cols = -age, names_to = "year", values_to = "lower_95") |>
      mutate(year = as.integer(year))
    df_upper <- as.data.frame(upper_rates)
    colnames(df_upper) <- forecasted_years
    df_upper$age <- ages
    upper_long <- df_upper |>
      pivot_longer(cols = -age, names_to = "year", values_to = "upper_95") |>
      mutate(year = as.integer(year))
    df_forecasted_long$lower_95 <- lower_long$lower_95
    df_forecasted_long$upper_95 <- upper_long$upper_95

    forecasted_results[[paste(i, j, sep = "_")]] <- df_forecasted_long
  }
}

# assemble fitted results (no uncertainty for in-sample)
final_fitted_df <- bind_rows(fitted_results)
final_fitted_df <- final_fitted_df |>
  transmute(geo = country, gender, year, age, mu = rate,
            total_var = NA_real_, aleatoric_var = NA_real_,
            epistemic_var = NA_real_, lower_95 = NA_real_,
            upper_95 = NA_real_)

# assemble forecast results with prediction intervals
final_forecasted_df <- bind_rows(forecasted_results)
final_forecasted_df <- final_forecasted_df |>
  transmute(geo = country, gender, year, age, mu = rate,
            total_var = ((upper_95 - lower_95) / (2 * 1.96))^2,
            aleatoric_var = NA_real_, epistemic_var = NA_real_,
            lower_95, upper_95)

# save fitted and forecasts
write.table(final_fitted_df, paste(path, "hu_fitted.csv", sep = "/"),
            sep = ",", col.names = FALSE, row.names = FALSE)
write.table(final_forecasted_df, paste(path, "hu_forecast.csv", sep = "/"),
            sep = ",", col.names = FALSE, row.names = FALSE)
