crime <- read.csv("http://datasets.flowingdata.com/crimeRatesByState-formatted.csv")
crime[1:6,]
library(aplpack)
faces(crime[,2:8], face.type = 1, labels=crime$state)
#without smile
crime_filled <- cbind(crime[,1:6], rep(0, length(crime$state)), crime[,7:8])
faces(crime_filled[,2:8], face.type = 1, fill = FALSE, scale = TRUE, labels=crime_filled$state)
