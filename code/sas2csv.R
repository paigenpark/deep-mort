library(rstudioapi)

# Getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
print( getwd() )

options(stringsAsFactors=F, scipen = 999)

pkg = 'Hmisc'
if (!require(pkg, character.only = TRUE)) {
  install.packages(pkg)
  library(pkg, character.only = TRUE)
}

setwd("../data/brfss_data/")

brfss <- sasxport.get("LLCP2015.XPT ")
write.csv(brfss, file = "brfss2015.csv")

brfss <- sasxport.get("LLCP2016.XPT ")
write.csv(brfss, file = "brfss2016.csv")

brfss <- sasxport.get("LLCP2017.XPT ")
write.csv(brfss, file = "brfss2017.csv")

brfss <- sasxport.get("LLCP2018.XPT ")
write.csv(brfss, file = "brfss2018.csv")

brfss <- sasxport.get("LLCP2019.XPT ")
write.csv(brfss, file = "brfss2019.csv")

brfss <- sasxport.get("LLCP2020.XPT ")
write.csv(brfss, file = "brfss2020.csv")

brfss <- sasxport.get("LLCP2021.XPT ")
write.csv(brfss, file = "brfss2021.csv")

brfss <- sasxport.get("LLCP2022.XPT ")
write.csv(brfss, file = "brfss2022.csv")
