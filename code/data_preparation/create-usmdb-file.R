# installing and loading packages 
library(here)
library(tidyverse)

# setting up path
path <- here("data", "us_lifetables", "States")
path

# read in data
### STATE DATA ###
# read in life tables for all 50 states 
states = c("AK", "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", 
           "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", 
           "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", 
           "WV", "WI", "WY")
f_data = list()
for (i in 1:length(states)){
  file = paste(states[i], "fltper_1x1.csv", sep = "_")
  f_data[[i]] = read.csv(paste(path, states[i], file, sep = "/"))
  f_data[[i]]$state = states[i]
}
names(f_data) <- states

m_data = list()
for (i in 1:length(states)){
  file = paste(states[i], "mltper_1x1.csv", sep = "_")
  m_data[[i]] = read.csv(paste(path, states[i], file, sep = "/"))
  m_data[[i]]$state = states[i]
}
names(m_data) <- states

# merge all states 
all_states_f <- do.call(rbind, f_data)
all_states_m <- do.call(rbind, m_data)
all_states <- rbind(all_states_f, all_states_m)
all_states_reduced <- all_states[, 1:5]

# setting up save_path
save_path <- here("data")
save_path

write.csv(all_states_reduced, paste(save_path, "usmdb.csv", sep = "/"), row.names = FALSE)




