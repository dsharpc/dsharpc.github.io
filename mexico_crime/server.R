#server.R


function(input, output){
  
##PESTAÑA DE MAPA
  # Mapa de la pestaña mapa
  output$mapadash1 <- renderLeaflet({
    leaflet() %>% addProviderTiles(providers$CartoDB.Positron) %>%  
      setView(lng = -99.1334, lat = 19.4331, zoom = 12)
     
  })
  
  observe({
    fecha_selec <- input$fecha
    data_crimen_f <- filter(data_crimen, (date >= fecha_selec[1]) & (date <= fecha_selec[2]) & (data_crimen$municipio %in% input$munic_sel))
    coord <- data_crimen %>% select(municipio, lat, long) %>% group_by(municipio) %>% summarise(latit = mean(lat), longi = mean(long))
    proxy <- leafletProxy("mapadash1", data = data_crimen_f)
    proxy %>% clearMarkers() %>% clearControls() %>%  setView(lng = -99.1334, lat = 19.4331, zoom = 12)
    
    proxy %>% addCircleMarkers(lng = ~long, lat = ~lat, radius = 5, fillOpacity = .75, color = ~ pal_cri(crime), label = ~crime) %>%
      setView(lng = coord[coord$municipio %in% input$munic_sel,]$longi, lat = coord[coord$municipio %in% input$munic_sel,]$latit, zoom = 13) #%>%
    #addLegend("bottomright", pal = pal_cri, values = ~crime, title = "Crímenes")
    
  })
  
  output$tabla <- renderTable(
      data_crimen %>% select(date, municipio, crime) %>% filter(date >= input$fecha[1] & date <= input$fecha[2] & municipio %in% input$munic_sel) %>%
      group_by(crime) %>% count(crime) %>% arrange(desc(n)) 
    )
  
  output$pcap <- renderInfoBox({
    crimen <- data_crimen %>% filter(date >= input$fecha[1] & date <= input$fecha[2], municipio %in% input$munic_sel) %>% nrow()
    popul <- data_crimen %>% filter(date >= input$fecha[1] & date <= input$fecha[2], municipio %in% input$munic_sel) %>% 
      select(municipio, population, cuadrante) %>% group_by(municipio, cuadrante) %>% summarise(poblacion = mean(population)) %>% 
      group_by(municipio) %>% summarise(total = sum(poblacion))
    infoBox(
      title = "Crímen per capita", 
      percent(crimen / popul[popul$municipio %in% input$munic_sel,]$total),
      icon = icon("percent"), color = "blue"
    )
    
  })

## PESTAÑA DE DASHBOARD 
  # Mapa de la pestaña dashboard
  output$mapadash2 <- renderLeaflet({
    leaflet() %>% addProviderTiles(providers$CartoDB.Positron) %>%
      setView(lng = -99.1334, lat = 19.4331, zoom = 10)
  })
  
  # Agregar y eliminar los marcadores por tipo de crimen
  observe({
    fecha_selec <- input$fecha
    crime_selec <- input$delitos_sel
    data_crimen_f <- filter(data_crimen, (date >= fecha_selec[1]) & (date <= fecha_selec[2]) & (crime %in% crime_selec))
    proxy <- leafletProxy("mapadash2", data = data_crimen_f)
    proxy %>% clearMarkers() %>% clearControls()
    proxy %>% addCircleMarkers(lng = data_crimen_f$long, lat = data_crimen_f$lat, color = ~pal_cri(crime), radius = 5, fillOpacity = .75) %>%
      addLegend("bottomright", pal = pal_cri, values = ~crime_selec, title = "Crímenes")
  })
  
  output$tot_box <- renderInfoBox({
    infoBox(
      "Total de crímenes", data_crimen %>% filter(date >= input$fecha[1] & date <= input$fecha[2], crime %in% input$delitos_sel) %>%
        nrow(),
      icon = icon("bar-chart"), color = "blue"
    )
  })
  
  output$maxmunic <- renderInfoBox({
    infoBox(
      "Municipio más crímen", data_crimen %>% filter(date >= input$fecha[1] & date <= input$fecha[2], crime %in% input$delitos_sel) %>%
        select(municipio) %>% group_by(municipio) %>% count %>% ungroup %>% slice(which.max(n)),
      icon = icon("map-marker"), color = "red"
    )
  })
  
  output$minmunic <- renderInfoBox({
    infoBox(
      "Municipio menos crímen", data_crimen %>% filter(date >= input$fecha[1] & date <= input$fecha[2], crime %in% input$delitos_sel) %>%
        select(municipio) %>% group_by(municipio) %>% count %>% ungroup %>% slice(which.min(n)) %>% na.omit,
      icon = icon("map-marker"), color = "green"
    )
  })
  
  output$crimcap <- renderPlot({
    data_crimen_pc <- data_crimen %>% select(municipio, year) %>% group_by(municipio, year) %>% count(municipio) %>% 
      filter(year == input$cpca) %>% left_join(poblacion) %>% mutate(CrimenPerCapita = n/total) %>% select(municipio, CrimenPerCapita) %>%
      arrange(desc(CrimenPerCapita)) %>% na.omit
    
    ggplot(data = data_crimen_pc) + 
      geom_bar(aes(x = reorder(municipio, -CrimenPerCapita), y = percent(CrimenPerCapita)), stat = "identity") + labs(y ="Crímen per capita", x = "Municipio") +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
    
  })
  
## PESTAÑA DE LÍNEA DE TIEMPO
  output$evotot <- renderPlot({
    if(input$tframe_l == "Por año"){
      data_crimen_l <- data_crimen %>% select(crime, year) %>% group_by(crime) %>% count(date = year)
    }else if(input$tframe_l == "Por mes"){
      data_crimen_l <- data_crimen %>% select(crime, date) %>%
        group_by(crime, date = floor_date(date, "month")) %>% count(date)
    }else{
     data_crimen_l <- data_crimen %>% group_by(crime) %>% count(date) 
    }
    
    ggplot()  + geom_line(aes(x = date, y = n), data = data_crimen_l[data_crimen_l$crime %in% input$comp1,]) + labs(x = "Fecha", y = "Número de delitos") + 
      geom_smooth(aes(x = date, y = n), data = data_crimen_l[data_crimen_l$crime %in% input$comp1,], colour = "blue") + 
      geom_line(aes(x = date, y = n), data = data_crimen_l[data_crimen_l$crime %in% input$comp2,]) + 
      geom_smooth(aes(x = date, y = n), data = data_crimen_l[data_crimen_l$crime %in% input$comp2,], colour = "red")
  })
  
  
  
## PESTAÑA DE ANÁLISIS POR CRIMEN
  output$plotmost <- renderPlot({
    fecha_selec <- input$fecha
    
    ggplot(data = filter(count(data_crimen, crime, date), crime %in% input$crimsel_sel)) + 
      geom_line(aes(x = date, y = n)) + xlim(fecha_selec[1], fecha_selec[2]) +
      ylim(0, max(filter(count(data_crimen, crime, date), crime %in% input$crimsel_sel)$n) + 1) +
      labs(x = "Fecha", y = "Número de delitos") + geom_smooth(aes(x = date, y = n), color = "red")
           })
  
  output$plotinci <- renderPlot({
    fecha_selec <- input$fecha
    ggplot(data = filter(count(data_crimen, crime, date, municipio), crime %in% input$crimsel_sel,
                         between(date, fecha_selec[1], fecha_selec[2]))) + 
      geom_bar(aes(x = municipio, y = n), stat = "identity") + 
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      labs(x = "Municipio", y = "Número de delitos")
  })
  
  output$plottime <- renderPlot({
    ggplot(data = filter(count(data_crimen, crime, hour), crime %in% input$crimsel_sel)) + 
      geom_jitter(aes(x = hour, y = n), color = "blue") + labs(y ="Número de delitos", x = "Hora")
    
  })
  
  
}
