#ui.R

#Diseño de la barra lateral de la interfaz
sidebar <- dashboardSidebar(
    sidebarMenu(
    menuItem("Dashboard General", tabName = "dashboard", icon = icon("dashboard")),
    menuItem("Análisis Espacial", tabName = "mapa", icon = icon("map")),
    menuItem("Línea de tiempo", tabName = "timeline", icon = icon("line-chart")),
    menuItem("Análisis por delito", tabName = "por_delito", icon = icon("exclamation-circle")),
    br(),
    dateRangeInput(inputId = "fecha", label = "Selecciona el rango de fechas de tu interés", 
                   start = max(data_crimen$date, na.rm = TRUE) - 30, end = max(data_crimen$date, na.rm = TRUE),
                   min = min(data_crimen$date, na.rm = TRUE), max = max(data_crimen$date, na.rm = TRUE))
    )
  )

#Diseño de los elementos dentro del cuadro principal en la interfaz
body <- dashboardBody(
    tabItems(
      #Contenido para Dashboard General
      tabItem(tabName = "dashboard",
                h2("Información general de los delitos"),
              fluidRow(
                infoBoxOutput("tot_box"),
                infoBoxOutput("maxmunic"),
                infoBoxOutput("minmunic")
              ),
              
              fluidRow(
                column(4,
                       box(checkboxGroupInput(inputId = 'delitos_sel', "Selecciona los crímenes a visualizar:", 
                                          as.vector(unique(data_crimen$crime))), width = 15)
                       ),
                column(8,
                leafletOutput(outputId = 'mapadash2', width = "100%", height = 540)
                )
              ),
              fluidRow(
                       selectInput(inputId = "cpca", "Selecciona el año para ver los crímenes per capita:", c(2013, 2014, 2015))
              ),
              fluidRow(
                       box(title = "Crímen per capita (# de crímenes / población del municipio", width = 12, plotOutput(outputId = "crimcap"))
              )
              
              
      ),
      
      #Contenido para tab de mapa
      tabItem(tabName = "mapa",
              h2("Mapa de la Ciudad de México"),
              fluidRow(
                selectInput(inputId = 'munic_sel', "Selecciona el municipio para visualizar sus crímenes:", 
                            as.vector(na.omit(unique(data_crimen$municipio)))),
                infoBoxOutput("pcap")
                ),
              fluidRow(
                column(width = 4,
                       tableOutput("tabla")),
                column(width = 8,
              leafletOutput(outputId = "mapadash1", width = "100%", height = 600)
             
                )
              )
      ),
      
      #Contenido para tab de línea de tiempo
      tabItem(tabName = "timeline",
              h2("Evolución de la criminalidad"),
              fixedRow(
                column(width = 4, offset = .2,
                  radioButtons(inputId = "tframe_l", label = "Selecciona la unidad de tiempo deseada", c("Por día", "Por mes", "Por año"))  
                ),
                column(width = 3,
                       box(title = "Selecciona un crimen a comparar", solidHeader = TRUE, 
                           selectInput(inputId = "comp1", "Crimen 1", choices = na.omit(unique(data_crimen$crime))), width = 15, background = "blue")),
                column(width = 3,
                       box(title = "Selecciona el otro crimen", solidHeader = TRUE,
                           selectInput(inputId = "comp2", "Crimen 2", choices = na.omit(unique(data_crimen$crime))), width = 15, background = "red"))
                
              ),
              
              fixedRow(
                column(12,
                       box(title = "Evolución total de crímenes", width = 12, plotOutput(outputId = "evotot"))
                       
                       )
                
                
                
                
              )
              
      ),
      
      #Contenido para tab de análisis por delito
      tabItem(tabName = "por_delito",
              fluidRow(
              h2("Análisis por delito cometido"),
              selectInput(inputId = "crimsel_sel", "Selecciona un crimen", unique(data_crimen$crime)),
              fluidRow(
              #selectInput(inputId = "munic_sel", "Selecciona un municipio", unique(na.omit(data_crimen$municipio))),
              box(title = "Evolución del crimen", width = 6, plotOutput(outputId = "plotmost")),
              box(title = "Incidencia por municipio", width = 6, plotOutput(outputId = "plotinci"))
              ),
              fluidRow(
                box(title = "Incidencia por hora (de 2013 a 2016)", width = 12, plotOutput(outputId = "plottime"))
              )
      )
  )
)
)

#Unión de los elementos del dashboard
dashboardPage(
  
  dashboardHeader(title= "Crímen en la CDMX"),
  sidebar,
  body
)
