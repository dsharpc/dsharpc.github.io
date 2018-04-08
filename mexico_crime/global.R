
#Verificar instalación de paquetes
if(!require(shiny, quietly = TRUE, warn.conflicts = FALSE) ){
  install.packages('shiny', 
                   dependencies = TRUE, 
                   repos = "http://cran.us.r-project.org")
}

if(!require(leaflet, quietly = TRUE, warn.conflicts = FALSE) ){
  install.packages('leaflet', 
                   dependencies = TRUE, 
                   repos = "http://cran.us.r-project.org")
}

if(!require(shinydashboard, quietly = TRUE, warn.conflicts = FALSE) ){
  install.packages('shinydashboard', 
                   dependencies = TRUE, 
                   repos = "http://cran.us.r-project.org")
}

if(!require(tidyverse, quietly = TRUE, warn.conflicts = FALSE) ){
  install.packages('tidyverse', 
                   dependencies = TRUE, 
                   repos = "http://cran.us.r-project.org")
}

if(!require(lubridate, quietly = TRUE, warn.conflicts = FALSE) ){
  install.packages('lubridate', 
                   dependencies = TRUE, 
                   repos = "http://cran.us.r-project.org")
}

if(!require(DT, quietly = TRUE, warn.conflicts = FALSE) ){
  install.packages('DT', 
                   dependencies = TRUE, 
                   repos = "http://cran.us.r-project.org")
}

if(!require(scales, quietly = TRUE, warn.conflicts = FALSE) ){
  install.packages('scales', 
                   dependencies = TRUE, 
                   repos = "http://cran.us.r-project.org")
}


# Carga de paquetes

library(shiny)
library(leaflet)
library(shinydashboard)
library(tidyverse)
library(lubridate)
library(DT)
library(scales)




#carga de datos desde compu (más rápido por ahorita)
#setwd("/Users/danielsharp/Google Drive/ITAM/Maestria Data Science/Semestre 1/Seminario - Programming for DS/R")
data_crimen <- read_csv("clean-data/crime-lat-long.csv")
data_cuadrantes <- read_csv("clean-data/cuadrantes-hoyodecrimen.csv")



# Carga datos de página de Diego Valle
#temp <- tempfile()
#download.file("https://data.diegovalle.net/hoyodecrimen/cuadrantes.csv.zip", temp, method = "auto")
#data_crimen <- read_csv(unzip(temp, "clean-data/crime-lat-long.csv"))
#data_cuadrantes <- read_csv(unzip(temp, "clean-data/cuadrantes-hoyodecrimen.csv"))
#unlink(temp)
#rm(temp)

data_crimen <- data_crimen %>% na.omit
data_crimen$crime <- as.factor(data_crimen$crime)

data_cuadrantes <- data_cuadrantes %>% select(cuadrante, municipio, population) %>% unique
data_crimen <- left_join(data_crimen, data_cuadrantes)
data_crimen <- data_crimen %>% na.omit
poblacion <- select(data_crimen, municipio, population) %>% unique %>% group_by(municipio) %>% summarise(total = sum(population))


#Paletas
pal_cri <- colorFactor(rainbow(19), data_crimen$crime %>% unique)
