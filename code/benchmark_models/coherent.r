##### Coherent Product-Ratio #####

# The fitting process here is more complex than the other benchmark models

# Because the data for some countries is shorter (less years available)
# we need to fit two rounds of the coherent model

# 1) A coherent product–ratio model is fit on all countries with mortality 
#    time series extending back to 1959 (this excludes five countries, but 
#    includes all years)
# 2) A second coherent model restricted to the common time window
#    across all countries is fit (this includes all countries -- except Germany
#    which was leading to fitting issues, but not all years)

# All forecasts are exported into the same csv file.  

# --------------------------------------------------------------------------------
# Note: The forecast() call below sometimes prints
# 
#   Error in stats::arima(x = …): non-stationary AR part from CSS
 
# Why does this occur?
#   auto.arima() searches many candidate ARIMA orders. Each one is first
#   estimated with conditional-sum-of-squares (“CSS”). If the provisional
#   AR coefficients are outside the stationarity region, stats::arima()
#   throws the message before control returns to R.  The call is wrapped
#   in try(), so execution continues.

#   The candidate that fails is discarded (its IC is set to Inf) and the
#   search moves on.  The winning order is ultimately re-fitted with full
#   maximum likelihood, so the final model is guaranteed stationary.

#   Good forecasts are still produced for every year. 
#
# Source: forecast::auto.arima() and helper myarima()  
# https://github.com/robjhyndman/forecast/blob/master/R/newarima2.R

# --------------------------------------------------------------------------------
library(demography)
library(tidyverse)
library(reshape2)
library(glue)
library(here)

# parameters
forecast_years    <- 2006:2015
n_iter            <- 5

# load and prepare data
path <- here("data")
file <- "country_training.txt"
country_training <- read.table(paste(path, file, sep = "/"), header = FALSE,
                               col.names = c("Country", "Gender", "Year", "Age", "Rate"))

country_training <- country_training |>
  arrange(Country, Gender, Year, Age)

# get all countries whose data does not extend back to 1959
excluded_countries = c()

for (country in unique(country_training$Country)) {
  one_country = country_training |>
    filter(Country == country) 
  
  year_min = min(one_country$Year, na.rm = TRUE)
  
  if (year_min > 1959) {
    excluded_countries = c(excluded_countries, country)
  }
}

ages_all <- sort(unique(country_training$Age))

# create function to build a list of demogdata objects 
# (one per country–gender combo)
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
        data  = mx_mat,
        pop   = Ext,
        ages  = ages_vec,
        years = years_i,
        type  = "mortality",
        label = c,
        name  = g
      )
      list_name <- glue("{c}_{g}")
      data_objects[[list_name]] <- data
    }
  }
  data_objects
}

# collapse list of demogdata objects into a single combined object
combine_demog_list <- function(dlist) {
  template <- dlist[[1]]
  template$rate <- lapply(dlist, function(x) x$rate[[1]])
  template$pop  <- lapply(dlist, function(x) x$pop[[1]])
  template
}


# prepare the datasets


# full-history set (problem countries removed)
full_hist_df <- country_training %>%
  filter(!Country %in% excluded_countries)
full_demog_list   <- build_demog_list(full_hist_df, ages_all)
combined_full     <- combine_demog_list(full_demog_list)

# restricted years set for all countries (except Germany - which was causing problems)
# find the intersection of years that every country–gender combo possesses
germany <- c(58, 59)
common_years <- country_training %>%
  group_by(Country, Gender) %>%
  summarise(yrs = list(unique(Year)), .groups = "drop") %>%
  pull(yrs) %>%
  reduce(intersect)

df_restricted      <- country_training %>% filter(Year %in% common_years,
                                                  !Country %in% germany)
demog_list_restricted  <- build_demog_list(df_restricted, ages_all)
combined_restricted      <- combine_demog_list(demog_list_restricted)


# fit, forecast, and merge into one file (repeat n_iter times)
for (iter in seq_len(n_iter)) {
  set.seed(iter)
  # Coherent fit on full‑history subset
  fit_full <- coherentfdm(combined_full)
  fc_full  <- forecast(fit_full, h = length(forecast_years))
  
  # Coherent fit on restricted years (data back to 1960)
  fit_restrict <- coherentfdm(combined_restricted)
  fc_restrict  <- forecast(fit_restrict, h = length(forecast_years))
  
  # Build a unified list of forecasts using only the population-level labels
  pop_labels <- union(names(full_demog_list),   # everything in the full model
                      names(demog_list_restricted))  
  
  fc_mixed_rates <- map(pop_labels, function(label) {
    country_id <- as.integer(str_split(label, "_", simplify = TRUE)[1])
    if (country_id %in% excluded_countries) {
      fc_restrict[[label]]$rate[[1]]
    } else {
      fc_full[[label]]$rate[[1]]
    }
  })
  names(fc_mixed_rates) <- pop_labels
  
  
  # convert to long data frame
  forecasted_results <- imap(fc_mixed_rates, function(mat, label) {
    df <- as.data.frame(mat)
    df$age <- ages_all
    long_df <- melt(df, id.vars = "age", variable.name = "year", value.name = "rate")
    long_df$year <- rep(forecast_years, each = length(ages_all))
    label_split <- str_split(label, "_", simplify = TRUE)
    long_df$country <- as.numeric(label_split[1])
    long_df$gender  <- as.numeric(label_split[2])
    long_df %>% select(country, gender, year, age, rate)
  })
  
  final_df <- bind_rows(forecasted_results)
  
  out_file <- glue("coherent_forecast_{iter}.csv")
  #write.table(final_df, paste(path, out_file, sep = "/"), sep = ",", col.names = FALSE, row.names = FALSE)
  
  print(glue("Iteration {iter} complete – saved to {out_file}"))
}

