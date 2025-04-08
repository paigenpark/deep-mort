library(demography)
library(tidyverse)
library(reshape2)

# sets working directory to the location of this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

country_training <- read.table("../../data/country_training.txt", header = FALSE)
countries <- unique(country_training[,1])
genders <- unique(country_training[,2])
years <- unique(country_training[,3])
ages <- unique(country_training[,4])
forecasted_years <- 2006:2015
colnames(country_training) <- c('Country', 'Gender', 'Year', 'Age', 'Rate')

data_objects <- list()
fitted_results <- list()
forecasted_results <- list()

for (i in countries) {
  for (j in genders) {
    filtered <- country_training |>
      filter(country_training[,1] == i & country_training[,2] == j)
    
    mx_mat <- unlist(lapply(years, function(i) filtered[filtered[,3] == i, 5]))
    mx_mat <- matrix(mx_mat, nrow = length(ages), ncol = length(years), byrow = FALSE)
    colnames(mx_mat) <- years
    mx_mat[is.na(mx_mat)] <- 9e-6
    mx_mat[mx_mat == 0] <- 9e-6
    
    Ext <- matrix(1, nrow = length(ages), ncol = length(years))
    
    data <- demogdata(
      data = mx_mat,
      pop = Ext,
      ages = ages,
      years = years,
      type = "mortality",
      label = i,
      name = j
    )

    list_name <- paste(i, j, sep = "_")
    data_objects[[list_name]] <- data
  }
}

combined_data <- data_objects[[1]]
combined_data$rate <- lapply(data_objects, function(x) x$rate[[1]])
combined_data$pop  <- lapply(data_objects, function(x) x$pop[[1]])

# fits
coherent_fits <- coherentfdm(combined_data)
plot(residuals(coherent_fits$product))
plot(residuals(coherent_fits$ratio$`87_0`))

# forecasts
coherent_forecasts <- forecast(coherent_fits, h=10, drange=c(0,0.5))

plot(coherent_forecasts$product, 'c', comp=2)
plot(coherent_fits$product$y, col='gray', ylim=c(-11,-0.5),
     main="Mortality forecasts product: 2010-2059")
lines(coherent_forecasts$product)

plot(coherent_forecasts$ratio$`87_0`, 'c', comp=2)
plot(coherent_fits$ratio$`87_0`$y,col='gray',
     main="Mortality forecasts ratio (M/F): 2010-2059")
lines(coherent_forecasts$ratio$`87_0`) 

# save forecasted results
forecasted_rates <- list()
forecasted_results <- list()
for (i in 1:76) {
  forecasted_rates[[i]] <- coherent_forecasts[[i]]$rate[[1]]
  
  df_forecasted <- as.data.frame(forecasted_rates[[i]])
  df_forecasted$age <- ages
  df_forecasted_long <- melt(df_forecasted, id.vars = "age", 
                             variable.name = "year",
                             value.name = "rate")
  df_forecasted_long$year <- rep(forecasted_years, each = length(ages))
  
  # Extract country and gender from names
  label_split <- strsplit(names(coherent_forecasts[i]), "_")[[1]]
  df_forecasted_long$country <- as.numeric(label_split[1])
  df_forecasted_long$gender <- as.numeric(label_split[2])

  forecasted_results[[i]] <- df_forecasted_long
}

final_forecasted_df <- bind_rows(forecasted_results)
final_forecasted_df <- final_forecasted_df |>
  select(country, gender, year, age, rate)  

# View structure of final datasets
head(final_fitted_df)
head(final_forecasted_df)

# uncomment to re-save prediction files
#write.table(final_fitted_df, "../../data/coherent_fitted_all.csv", sep=",", col.names = FALSE,
          #  row.names = FALSE)

#write.table(final_forecasted_df, "../../data/coherent_forecast_all.csv", sep=",", 
          #  col.names = FALSE, row.names = FALSE)



