library(dialectR)
library(ggplot2)
library(mgcv)
library(itsadug)

path_to_lstm_final_results = ""

bpp <- read.csv(path_to_lstm_final_results)
bpp <- as.data.frame(cbind(bpp$lang, bpp$avg_len, bpp$test_shannon))
colnames(bpp) <- c("lang", "avg_len", "lstm")
dutch_points <- get_points(system.file("extdata", "DutchKML.kml", package="dialectR"))

dutch <- merge(bpp, dutch_points, by.x="lang", by.y="name")
dutch$latitude <- as.numeric(dutch$latitude)
dutch$longitude <- as.numeric(dutch$longitude)

# distance from standard Dutch AND remove Belgium
avg_dist_to_std <- aggregate(dialectNL$PronDistStdDutch, list(dialectNL$Location), mean)
dutch <- dutch[dutch$lang %in% avg_dist_to_std$Group.1,]
dutch <- merge(dutch, avg_dist_to_std, by.x="lang", by.y="Group.1")

geo <- gam(as.numeric(lstm) ~ s(longitude, latitude), data = dutch)
summary(geo)
pvisgam(geo, view = c("longitude", "latitude"), select = 1, color = "topo", too.far = 0.05, add.color.legend=F, main=NULL, hide.label=T)
fvisgam(geo, plot.type = "contour", color = "topo", too.far = 0.05, view = c("longitude", "latitude"), main="", add.color.legend=F)

cor(as.numeric(dutch$lstm), as.numeric(dutch$avg_len), method="spearman") 
cor(as.numeric(dutch$lstm), as.numeric(dutch$avg_len), method="pearson")