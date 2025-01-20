library(tidyterra)
library(tidyverse) 
library(scales) 
library(terra)
library(ggplot2)
library(maptiles)
library(geodata)
library(sf)
library(maptiles)
library(svglite)
# 设置全局默认主题，所有 ggplot 都会使用
theme_set(
  theme(
    text = element_text(face = "bold"),  # 全局加粗
  )
)

# 设置工作目录为当前目录
setwd('F:/Homework/BVOC-SOA/CI/sounddata/数据库处理/地图绘制')
getwd()

# 读取上海行政区域
#Shanghaishp1 <- vect('上海行政区域县级.shp')
#Shanghaishp <- crop(Shanghaishp1, Shanghaishp2)
Shanghaishp <- vect('上海市_县界.shp')
# 将上海县级边界的AOI投影为 EPSG:3857
aoi_shanghai_proj <- project(Shanghaishp, "EPSG:3857")

apikey <- '830ad848-6e37-4a79-ba97-d66ae2558fc4'
# 获取瓦片图作为背景
rgb_tile <- get_tiles(aoi_shanghai_proj, 
                      crop = TRUE, 
                      zoom = 10, project = FALSE, cachedir = ".",
                      provider = "Stadia.Stamen.TonerLite",
                      apikey=apikey)

#rgb_tile <- get_tiles(aoi_shanghai_proj, crop = TRUE, 
                      #zoom = 10, project = FALSE, cachedir = ".",
                     # provider = "Esri.WorldImagery")



# 读取 NDVI 数据，并投影到与 AOI 相同的坐标系
NDVI <- rast('上海市_有途GIS.tif')  # NDVI 数据
NDVI_proj <- project(NDVI, crs(aoi_shanghai_proj))
NDVI_cropped <- mask(NDVI_proj, aoi_shanghai_proj)
NDVI_min <- minmax(NDVI_proj)[1]  # 获取最小值
NDVI_max <- minmax(NDVI_proj)[2]  # 获取最大值
NDVI_normalized <- (2 * (NDVI_cropped - NDVI_min) / (NDVI_max - NDVI_min)) - 1

# Sample data frame with coordinates and names
locations <- data.frame(
  name = c("Chongming", "Jinhai", "Zhongshan"),
  lat = c(31.700638, 31.253461, 31.224391),
  lon = c(121.355921, 121.648004, 121.414147)
)
# Convert to an sf object
locations_sf <- st_as_sf(locations, coords = c("lon", "lat"), crs = crs(aoi_shanghai_proj))


# Labels
cap_lab <- "Shanghai 2023-2024"
tit_lab <- "Experimental Site"
ggplot() +
  geom_spatraster_rgb(data = rgb_tile, alpha = 1) +
  geom_spatraster(data = NDVI_normalized, aes(fill = Band_1))+
  geom_spatvector(fill = NA) +
  geom_spatvector(data = aoi_shanghai_proj, fill = NA, color = "black", size = 0.5)+  
  scale_fill_princess_c(
    palette = "snow", alpha = 0.9,
    labels = scales::label_number(suffix = "")
  )+
  # scale_fill_whitebox_c(
  #   palette = "pi_y_g", alpha = 0.5,
  #   labels = scales::label_number(suffix = "")
  # ) +
  labs(title = "Map of Shanghai with Key Locations") +
  theme_minimal(base_size = 14) +  
  coord_sf(expand = FALSE) +
  labs(
    title = tit_lab,
    fill = "NDVI",
    caption = cap_lab
  )+ 
  theme(
    # Position the legend inside the plot at the bottom right
    legend.position = c(1, 0),  # x and y coordinates; 1, 0 for bottom right
    legend.justification = c(1, 0),  # Aligns the legend box to the bottom right of its position
    legend.box.just = "right",  # Aligns the contents of the legend box to the right
    legend.text.align = 0,  # Aligns the legend text to the left within the legend box
    legend.title = element_text(size = 10),  # Control the size of the legend title
  )
   
# Labels
cap_lab <- "Shanghai 2023-2024"
tit_lab <- "Experimental Site"
# Define your plot
plot <- ggplot() +
  geom_spatraster_rgb(data = rgb_tile, alpha = 1) +
  geom_spatraster(data = NDVI_normalized, aes(fill = Band_1)) +
  geom_spatvector(data = aoi_shanghai_proj, fill = NA, color = "black", size = 0.5) +
  scale_fill_whitebox_c(
    palette = "purple", alpha = 0.9,
    labels = scales::label_number(suffix = "")
  )+
  labs(
    title = tit_lab,
    fill = "NDVI",
    caption = cap_lab,
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(size = 12.5, face = "bold", color = "black", angle = 45, hjust = 1),  # x 轴刻度加粗+黑色
    axis.text.y = element_text(size = 12.5, face = "bold", color = "black"),  # y 轴刻度加粗+黑色
    text = element_text(family = "Times New Roman", face = "bold"),  # Set global font family
    legend.position = c(1, 0),
    legend.justification = c(1, 0),
    legend.box.just = "right",
    legend.text.align = 0,
    legend.title = element_text(size = 10),
  ) +
  coord_sf(expand = FALSE)

# Display the plot
print(plot)
# Save the plot as an SVG file
ggsave("Shanghai_NDVI.svg", plot = plot, device = "svg", width = 6, height = 6, units = "in")



