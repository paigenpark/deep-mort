# Define a vector of required packages
required_packages <- c("demography", "tidyverse", "reshape2", "glue", "here")

# Function to check and install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Apply the function to all required packages
invisible(lapply(required_packages, install_if_missing))

# Load the libraries
lapply(required_packages, library, character.only = TRUE)


path <- here("data")
country_training <- read.table(paste(path, "country_training.txt", sep = "/"), 
                              header = FALSE)
countries <- unique(country_training[,1])
genders <- unique(country_training[,2])
ages <- unique(country_training[,4])
forecasted_years <- 2006:2015
colnames(country_training) <- c('Country', 'Gender', 'Year', 'Age', 'Rate')

fitted_results <- list()
forecasted_results <- list()


for (iter in 1:5) {
  set.seed(iter)
  for (i in countries) {
    for (j in genders) {
      filtered <- country_training |>  
        filter(country_training[,1] == i & country_training[,2] == j) 
      # get number of years available for country/gender combo
      years <- sort(unique(filtered$Year))

      mx_df <- filtered |>
        pivot_wider(names_from = 'Year',
                    values_from = 'Rate') |>
        select(-Age, -Gender, -Country)
      mx_mat <- as.matrix(mx_df)
      #colnames(mx_mat) <- years
      mx_mat[mx_mat == 0 | is.na(mx_mat)] <- 9e-06
      
      # mx_mat <- unlist(lapply(years, function(yr) filtered[filtered[,3] == yr, 5]))
      # mx_mat <- matrix(mx_mat, nrow = length(ages), ncol = length(years), byrow = FALSE)
      # mx_mat[mx_mat == 0 | is.na(mx_mat)] <- 9e-06
      
      Ext <- matrix(1, nrow = nrow(mx_mat), ncol = ncol(mx_mat))
      
      data <- demogdata(
        data = mx_mat,
        pop = Ext,
        ages = ages,
        years = years,
        type = "mortality",
        label = i,
        name = j
      )
      
      lc_output <- lca(data,
                       years = years,
                       ages = ages,
                       adjust = 'none')
      
      fitted <- exp(lc_output$fitted$y)
      df_fitted <- as.data.frame(fitted)
      df_fitted$age <- ages
      df_fitted_long <- melt(df_fitted, id.vars = "age", 
                             variable.name = "year",
                             value.name = "rate")
      df_fitted_long$year <- rep(years, each = length(ages))
      df_fitted_long$country <- i
      df_fitted_long$gender <- j
      fitted_results[[paste(i, j, sep = "_")]] <- df_fitted_long
        
      forecasted <- forecast(lc_output, h=10) 
      forecasted_rates <- do.call(cbind, forecasted$rate[1])
      df_forecasted <- as.data.frame(forecasted_rates)
      df_forecasted$age <- ages
      df_forecasted_long <- melt(df_forecasted, id.vars = "age", 
                             variable.name = "year",
                             value.name = "rate")
      df_forecasted_long$year <- rep(forecasted_years, each = length(ages))
      df_forecasted_long$country <- i
      df_forecasted_long$gender <- j
      forecasted_results[[paste(i, j, sep = "_")]] <- df_forecasted_long
    }
  }
  
  final_fitted_df <- bind_rows(fitted_results)
  final_fitted_df <- final_fitted_df |>
    select(country, gender, year, age, rate)
  final_forecasted_df <- bind_rows(forecasted_results)
  final_forecasted_df <- final_forecasted_df |>
    select(country, gender, year, age, rate)  

  # uncomment to re-save prediction files
  write.table(final_forecasted_df, paste(path, glue("lc_forecast_{iter}.csv"), sep = "/"), 
              sep=",", col.names = FALSE,
              row.names = FALSE)

  print(glue("Iteration {iter} complete – saved to lc_forecast_{iter}.csv"))
}


