library(tidyverse)
library(ggtext)
library(ggdist)
library(glue)
library(patchwork)
library(readxl)
library(latex2exp)
library(scico)
library(extrafont)
library(readr)
library(reticulate)
library(ggplot2)
library(MetBrewer)  



# 设置全局默认主题，所有 ggplot 都会使用
theme_set(
    theme(
      text = element_text(face = "bold"),  # 全局加粗
    )
)

# 设置工作目录为当前目录
setwd('F:/Homework/BVOC-SOA/CI/sounddata/数据库处理')
getwd()

# 读取数据
data <- read_csv("BPMvocieclass.csv")
data$class <- as.factor(data$class)
# 重命名列 tempo 为 BPM
data <- data %>% rename(BPM = tempo)
# 查看 BPM 列的取值范围
range(data$BPM, na.rm = TRUE)


plotstat=function(data,variable,variablecn,ylimx,ylimy,labelpositionx){
  
  
  
# 设置背景颜色
bg_color <- "grey97"
variablee=variable



mean_price <- mean(pull(select(data,variablee)), na.rm = TRUE)
median_price <- median(pull(select(data,variablee)), na.rm = TRUE)
std_price <- sd(pull(select(data,variablee)), na.rm = TRUE)
n_rental_posts <- nrow(subset(data, !is.na(variable)))


p <- data %>% 
  ggplot(aes(class, !!sym(variablee))) +
  # stat_halfeye(fill_type = "segments", alpha = 0.3,side = "left",fill="#B3CAD8")+
  # stat_histinterval(slab_color = "grey60", outline_bars = TRUE,side = "right",fill = "#97A5C0") +
  # 上部 slabinterval
  stat_slabinterval( aes(thickness = after_stat(pdf)), side = "right", slab_fill = "#FA5D63", 
    interval_color = "grey0", slab_alpha = 0.95, 
    density = "bounded"
  ) +
  # stat_interval() +
  stat_summary(geom = "point", fun = median) +
  annotate("text", x = labelpositionx, y =ylimx, label = paste("Mean",variable), size = 5, hjust = 0.5, family = "Times New Roman") +
  stat_summary(
    aes(y = !!sym(variable)),
    geom = "text",
    fun.data = function(x) {
      data.frame(
        y = ylimx,
        label = sprintf("(%s)", scales::number(mean(ifelse(x > 0, x, NA), na.rm = TRUE), accuracy = 0.1)))}, size = 5
    , family = "Times New Roman") +
  geom_hline(yintercept = median_price, col = "grey30", lty = "dashed") +
  annotate("text", x = labelpositionx, y = median_price, label = paste("Median",variable), size = 5, hjust = 0, family = "Times New Roman") +
  # scale_x_discrete(labels = chinese_labels) +
  scale_y_continuous(breaks = seq(ylimx, ylimy, (ylimy-ylimx)/4),
                     labels = function(x) sprintf("%.2f", x)) +
  scale_color_manual(values = MetBrewer::met.brewer("Hokusai2")) +
  coord_flip(ylim = c(ylimx, ylimy), clip = "off") +
  guides(col = "none") +
  labs(
    title = "",
    x = NULL,
    y = "BPM")+
  theme(
    plot.background = element_blank(),
    panel.background = element_blank(),  # 设置面板背景为空白
    panel.grid = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.2, color = "grey90"),
    plot.title.position = "plot",
    plot.title = element_text(hjust = 0.5,size = 17, family = "Times New Roman"),
    axis.text.y = element_text(hjust = 1, margin = margin(r = 10)),
    plot.margin = margin(4, 4, 4, 4),
    axis.text = element_text(size = 14, family = "Times New Roman"), 
    axis.title = element_text(size = 14, family = "Times New Roman")
  )


return(p)
}


jpeg("声音种类BPM.jpg", width = 5000, height =6500,
     units = "px", res = 800, family="Times New Roman")
plotstat(data,"BPM","BPM",35,245,10.5)
dev.off() 



# 读取数据
data <- read_csv("BPMbirdclass.csv")
data$habitat <- as.factor(data$habitat)
# 重命名列 tempo 为 BPM
data <- data %>% rename(BPM = tempo)
# 查看 BPM 列的取值范围
range(data$BPM, na.rm = TRUE)


plotstat=function(data,variable,variablecn,ylimx,ylimy,labelpositionx){
  
  
  
# 设置背景颜色
bg_color <- "grey97"
variablee=variable

mean_price <- mean(pull(select(data,variablee)), na.rm = TRUE)
median_price <- median(pull(select(data,variablee)), na.rm = TRUE)
std_price <- sd(pull(select(data,variablee)), na.rm = TRUE)
n_rental_posts <- nrow(subset(data, !is.na(variable)))


p <- data %>% 
  ggplot(aes(habitat, !!sym(variablee))) +
  stat_halfeye(fill_type = "segments", alpha = 0.3,side = "left",fill="#ADB5AB")+
  stat_histinterval(slab_color = "grey60", outline_bars = TRUE,side = "right",fill = "#678F74") +
  stat_interval() +
  stat_summary(geom = "point", fun = median) +
  annotate("text", x = labelpositionx, y =ylimx, label = paste("Mean",variable), size = 5, hjust = 0.5, family = "Times New Roman") +
  stat_summary(
    aes(y = !!sym(variable)),
    geom = "text",
    fun.data = function(x) {
      data.frame(
        y = ylimx,
        label = sprintf("(%s)", scales::number(mean(ifelse(x > 0, x, NA), na.rm = TRUE), accuracy = 0.1)))}, size = 5
    , family = "Times New Roman") +
  geom_hline(yintercept = median_price, col = "grey30", lty = "dashed") +
  annotate("text", x = labelpositionx, y = median_price, label = paste("Median",variable), size = 5, hjust = 0, family = "Times New Roman") +
  # scale_x_discrete(labels = chinese_labels) +
  scale_y_continuous(breaks = seq(ylimx, ylimy, (ylimy-ylimx)/4),
                     labels = function(x) sprintf("%.2f", x)) +
  scale_color_manual(values = MetBrewer::met.brewer("VanGogh3")) +
  coord_flip(ylim = c(ylimx, ylimy), clip = "off") +
  guides(col = "none") +
  labs(
    title = "",
    x = NULL,
    y = "BPM")+
  theme(
    plot.background = element_blank(),
    panel.background = element_blank(),  # 设置面板背景为空白
    panel.grid = element_blank(),
    panel.grid.major.x = element_line(linewidth = 0.1, color = "grey75"),
    plot.title.position = "plot",
    plot.title = element_text(hjust = 0.5,size = 17, family = "Times New Roman"),
    axis.text.y = element_text(hjust = 1, margin = margin(r = 10)),
    plot.margin = margin(4, 4, 4, 4),
    axis.text = element_text(size = 14, family = "Times New Roman"), 
    axis.title = element_text(size = 14, family = "Times New Roman")
  )


return(p)
}
jpeg("鸟类BPM.jpg", width = 5000, height =4000,
     units = "px", res = 800, family="Times New Roman")
plotstat(data,"BPM","BPM",35,245,6.5)
dev.off() 


rent_title_words = read_csv("rent_title_words.csv")
# create the dataframe for the legend (inside plot)
df_for_legend <- rent_title_words %>% 
  filter(word == "beautiful")

p_legend <- df_for_legend %>% 
  ggplot(aes(word, price)) +
  stat_halfeye(fill_type = "segments", alpha = 0.3,side = "left",fill="#C3AAA6")+
  stat_histinterval(slab_color = "grey40", outline_bars = TRUE,side = "right",fill = "#DAC1C2") +
  stat_interval() +
  stat_summary(geom = "point", fun = median) +
  annotate(
    "text",
    x = c(0.8, 0.8, 0.8, 1.4, 1.8),
    y = c(1000, 5000, 3000, 2400, 4000),
    label = c("50 % of BPM", "95 % of BPM", 
              "80 % of BPM",
              "Median", "Distribution of BPM"), size = 7, vjust = 1,
  ) +
  geom_curve(
    data = data.frame(
      x = c(0.7, 0.80, 0.80, 1.225, 1.8),
      xend = c(0.95, 0.95, 0.95, 1.075, 1.8), 
      y = c(1800, 5000, 3000, 2300, 3800),
      yend = c(1800, 5000, 3000, 2100, 2500)),
    aes(x = x, xend = xend, y = y, yend = yend),
    stat = "unique", curvature = 0.2, size = 0.2, color = "grey12",
    arrow = arrow(angle = 20, length = unit(1, "mm"))
  ) +
  scale_color_manual(values = rev(MetBrewer::met.brewer("Greek"))) +
  coord_flip(xlim = c(0.75, 1.3), ylim = c(0, 6000), expand = TRUE) +
  guides(color = "none") +
  labs(title = "") +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5),
    plot.background = element_blank(),
    panel.background = element_blank(),  # 设置面板背景为空白
    panel.grid = element_blank(),  # 删除网格线
    axis.text.x = element_blank(),  # 删除x轴的文本
    axis.text.y = element_blank(),  # 删除y轴的文本
    axis.title.x = element_blank(),  # 删除x轴标题
    axis.title.y = element_blank(),  # 删除y轴标题
    axis.ticks = element_blank(),  # 删除刻度线
    axis.line = element_blank()  # 删除坐标轴线
  )

jpeg("分布图图例.jpg", width = 5500, height =5500,
     units = "px", res = 800, family="TimesSong")
print(p_legend)
dev.off() 


# 指定要使用的 Conda 环境
use_condaenv("dataanalysis", required = TRUE)
# 导入 Python 的 pickle 模块
py_run_string("import pickle")
# 读取 .pkl 文件
pkl_data <- py_run_string("with open('BPMvocieclass.pkl', 'rb') as f:
                              data = pickle.load(f)")
# 提取数据到 R 中
data <- pkl_data$data
rm(pkl_data)

range(unlist(data$Onsets), na.rm = TRUE)
data_expanded <- data %>%
  select(class, Onsets) %>%  # 选择只需要的列
  unnest(Onsets)  # 展开 Onsets 列
rm(data)

data_expanded <- data_expanded %>%
  mutate(Onsets_bin = cut(Onsets, breaks = seq(0, 60, by = 0.5), right = FALSE, include.lowest = TRUE))

# 计算每个 class 在不同 Onsets 分段下的占比
data_proportion <- data_expanded %>%
  group_by(class, Onsets_bin) %>%
  summarise(count = n()) %>%
  ungroup() %>%
  group_by(class) %>%
  mutate(proportion = count / sum(count))
rm(data_expanded)

# 使用 gsub 提取 Onsets_bin 的区间上下限，处理所有括号类型
data_proportion <- data_proportion %>%
  mutate(
    # 提取下限，保持一致
    lower_bound = as.numeric(gsub("\\[([0-9]+\\.?[0-9]*),.*", "\\1", Onsets_bin)),
    
    # 分别处理以 ")" 和 "]" 结尾的情况
    upper_bound = case_when(
      grepl("\\)$", Onsets_bin) ~ as.numeric(gsub(".*,([0-9]+\\.?[0-9]*)\\)$", "\\1", Onsets_bin)),  # 处理以 ")" 结尾的情况
      grepl("\\]$", Onsets_bin) ~ as.numeric(gsub(".*,([0-9]+\\.?[0-9]*)\\]$", "\\1", Onsets_bin))   # 处理以 "]" 结尾的情况
    ),
    
    # 计算区间中点
    Onsets_mid = (lower_bound + upper_bound) / 2
  )


# 计算各个 class 和 Onsets_mid 的密度
data_density <- data_proportion %>%
  group_by(class) %>%
  mutate(density = proportion / sum(proportion)) %>%  # 归一化密度值
  ungroup()

# 计算 KL 散度，与均匀分布比较
data_kl <- data_density %>%
  group_by(class) %>%
  summarise(
    kl_divergence = sum(density * log(density / (1/length(unique(Onsets_mid)))))
  ) %>%
  ungroup()

# 计算 KL 散度的最大可能值（当所有质量集中在一个 bin 时）
num_bins <- length(unique(data_density$Onsets_mid))  # 分段数
max_kl_divergence <- log(num_bins)  # KL 散度的最大值
min_kl_divergence <- 0  # KL 散度的最小值为 0（均匀分布）

# 计算 KL 散度的最大值和最小值
max_kl_divergence <- max(data_kl$kl_divergence)
min_kl_divergence <- min(data_kl$kl_divergence)

# 归一化 KL 散度（使用计算得出的最大最小值）
data_kl <- data_kl %>%
  mutate(normalized_kl = (kl_divergence - min_kl_divergence) / (max_kl_divergence - min_kl_divergence))

# 将归一化 KL 散度合并回原数据
data_density <- data_density %>%
  left_join(data_kl, by = "class")

# 在生成热图前过滤掉 NA 行
data_density <- data_density %>%
  filter(!is.na(class))


armyrose_colors <- c("#B95C7D","#D39DA8", "#F5D4D0", "white", "#D5D8A6", "#9CAC4F")
jpeg("Onsets声类.jpg", width = 6000, height =4000,
     units = "px", res = 800, family="Times New Roman")
ggplot(data_density, aes(x = Onsets_mid, y = class, fill = density)) +
  geom_tile(color = NA, alpha = 1) +  
  scale_fill_gradientn(
    colours = armyrose_colors,  # 使用 ArmyRose 配色
    name = "Density"
  )+
  # scale_fill_gradientn(colours = rev(met.brewer("Benedictus")), name = "Density") 
  scale_x_continuous(breaks = seq(0, 60, by = 5)) +
  scale_y_discrete(drop = TRUE) +  # 确保 y 轴不会显示无数据的 NA 标签
  labs(
    title = "",
    x = "Onsets (Midpoint of Bins)",
    y = ""
  ) +
  # 使用 Times New Roman 字体
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 14, face = "bold"),  # 设置字体
    plot.title = element_text(hjust = 0.5, size = 18),  # 标题居中
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.text = element_text(size = 12),
    plot.margin = margin(10, 80, 10, 10) , # 增加右边距以显示 KL 散度标注
    panel.grid = element_blank() , # 去除网格线
    axis.text.y = element_text(size = 12, color = "black", face = "bold"),
    axis.text.x = element_text(size = 12, color = "black", face = "bold"),
  ) +
  # KL 散度(Scale) 标签放在顶部
  annotate("text", x = 62, y = Inf, label = "KL Divergence\n(Scale)", 
           hjust = 0, family = "Times New Roman", size = 5, fontface = "bold", vjust = 0) +  # vjust 控制顶部位置
  # 在 KL 散度列外部标注 KL 散度
  geom_text(data = data_kl %>% filter(!is.na(normalized_kl)), aes(x = 62, y = class, label = round(normalized_kl, 2)),
            inherit.aes = FALSE, hjust = 0, family = "Times New Roman", size = 5, fontface = "bold") +
  coord_cartesian(xlim = c(0, 62), clip = "off")  # 扩展 x 轴范围，允许标注超出图的范围
dev.off() 




library(gridExtra)
library(scales)
library(ggdendro)

# 柱状图数据
bar_data <- data_kl %>%
  arrange(desc(normalized_kl))  # 排序
# 层次聚类数据准备
hc_data <- bar_data %>%
  column_to_rownames("class")  # 将 class 作为行名
dist_matrix <- dist(hc_data)  # 计算距离矩阵
hc <- hclust(dist_matrix)  # 层次聚类
dendro <- as.dendrogram(hc)
dendro_data <- ggdendro::dendro_data(dendro)


# 绘制层次聚类图
dendrogram <- ggplot(dendro_data$segments) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), linewidth = 0.8, color = "black") +
  theme_minimal() +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank(),
    text = element_text(family = "Times New Roman", size = 16, face = "bold"),
    plot.margin = margin(5, 5, 0, 5)  # 减少底部边距
  ) +
  labs(title = "Hierarchical Clustering of Classes")

# 绘制柱状图
bar_chart <- ggplot(bar_data, aes(x = reorder(class, -normalized_kl), y = normalized_kl)) +
  geom_bar(stat = "identity", fill = "#D39DA8", alpha = 0.8) +
  geom_text(
    aes(label = class, y = 0.05),  # 类别标签位置
    angle = 90, hjust = 0, size = 7, fontface = "bold", family = "Times New Roman"
  ) +
  geom_text(
    aes(label = scales::percent(normalized_kl, accuracy = 0.1), y = -0.01),  # KL 值标注
    size = 5, fontface = "bold", family = "Times New Roman", color = "black",
    angle = 45
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    x = NULL,  # 去掉 x 轴标签
    y = "Normalized KL Divergence"
  ) +
  theme_minimal() +coord_cartesian(clip = "off")+
  theme(
    text = element_text(family = "Times New Roman", size = 18, face = "bold"),  # 加大所有字体
    axis.text.x = element_blank(),  # 去掉 x 轴刻度
    axis.ticks.x = element_blank(),  # 去掉 x 轴刻度线
    axis.text.y = element_text(size = 16, face = "bold"),  # 增大 ytick 字体
    axis.title.y = element_text(size = 20, face = "bold"),  # 增大 ylabel 字体
    panel.grid = element_blank(),
    plot.margin = margin(0, 5, 5, 5)
  )

# 组合图
jpeg("Bar_and_Clustering_Modified.jpg", width = 6000, height = 5000, units = "px", res = 800)  # 降低图像高度

grid.arrange(
  dendrogram, bar_chart,
  heights = c(1, 3.5),  # 压缩层次聚类和柱状图的比例
  ncol = 1
)

dev.off()





# 导入 Python 的 pickle 模块
py_run_string("import pickle")
# 读取 .pkl 文件
pkl_data <- py_run_string("with open('BPMbirdclass.pkl', 'rb') as f:
                              data = pickle.load(f)")
# 提取数据到 R 中
data <- pkl_data$data
rm(pkl_data)

range(unlist(data$Onsets), na.rm = TRUE)
data_expanded <- data %>%
  select(habitat, Onsets) %>%  # 选择只需要的列
  unnest(Onsets)  # 展开 Onsets 列
rm(data)

data_expanded <- data_expanded %>%
  mutate(Onsets_bin = cut(Onsets, breaks = seq(0, 60, by = 0.5), right = FALSE, include.lowest = TRUE))

# 计算每个 class 在不同 Onsets 分段下的占比
data_proportion <- data_expanded %>%
  group_by(habitat, Onsets_bin) %>%
  summarise(count = n()) %>%
  ungroup() %>%
  group_by(habitat) %>%
  mutate(proportion = count / sum(count))
rm(data_expanded)

# 使用 gsub 提取 Onsets_bin 的区间上下限，处理所有括号类型
data_proportion <- data_proportion %>%
  mutate(
    # 提取下限，保持一致
    lower_bound = as.numeric(gsub("\\[([0-9]+\\.?[0-9]*),.*", "\\1", Onsets_bin)),
    
    # 分别处理以 ")" 和 "]" 结尾的情况
    upper_bound = case_when(
      grepl("\\)$", Onsets_bin) ~ as.numeric(gsub(".*,([0-9]+\\.?[0-9]*)\\)$", "\\1", Onsets_bin)),  # 处理以 ")" 结尾的情况
      grepl("\\]$", Onsets_bin) ~ as.numeric(gsub(".*,([0-9]+\\.?[0-9]*)\\]$", "\\1", Onsets_bin))   # 处理以 "]" 结尾的情况
    ),
    
    # 计算区间中点
    Onsets_mid = (lower_bound + upper_bound) / 2
  )


# 计算各个 class 和 Onsets_mid 的密度
data_density <- data_proportion %>%
  group_by(habitat) %>%
  mutate(density = proportion / sum(proportion)) %>%  # 归一化密度值
  ungroup()

# 计算 KL 散度，与均匀分布比较
data_kl <- data_density %>%
  group_by(habitat) %>%
  summarise(
    kl_divergence = sum(density * log(density / (1/length(unique(Onsets_mid)))))
  ) %>%
  ungroup()

# 计算 KL 散度的最大可能值（当所有质量集中在一个 bin 时）
num_bins <- length(unique(data_density$Onsets_mid))  # 分段数
max_kl_divergence <- log(num_bins)  # KL 散度的最大值
min_kl_divergence <- 0  # KL 散度的最小值为 0（均匀分布）

# 计算 KL 散度的最大值和最小值
max_kl_divergence <- max(data_kl$kl_divergence)
min_kl_divergence <- min(data_kl$kl_divergence)

# 归一化 KL 散度（使用计算得出的最大最小值）
data_kl <- data_kl %>%
  mutate(normalized_kl = (kl_divergence - min_kl_divergence) / (max_kl_divergence - min_kl_divergence))

# 将归一化 KL 散度合并回原数据
data_density <- data_density %>%
  left_join(data_kl, by = "habitat")

# 在生成热图前过滤掉 NA 行
data_density <- data_density %>%
  filter(!is.na(habitat))


jpeg("Onsets鸟类.jpg", width = 6000, height =4000,
     units = "px", res = 800, family="Times New Roman")
ggplot(data_density, aes(x = Onsets_mid, y = habitat, fill = density)) +
  geom_tile(color = NA, alpha = 0.8) +  # 热图边界模糊化（去掉边界颜色，增加透明度）
  scale_fill_gradientn(colours = rev(met.brewer("Benedictus")), name = "Density") +  # 使用反转的 MetBrewer 的 Benedictus 调色板
  scale_x_continuous(breaks = seq(0, 60, by = 5)) +
  scale_y_discrete(drop = TRUE) +  # 确保 y 轴不会显示无数据的 NA 标签
  labs(
    title = "",
    x = "Onsets (Midpoint of Bins)",
    y = ""
  ) +
  # 使用 Times New Roman 字体
  theme_minimal() +
  theme(
    text = element_text(family = "Times New Roman", size = 14,face = "bold"),  # 设置字体
    plot.title = element_text(hjust = 0.5, size = 18),  # 标题居中
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    axis.text = element_text(size = 12),
    plot.margin = margin(10, 80, 10, 10) , # 增加右边距以显示 KL 散度标注
    panel.grid = element_blank()  # 去除网格线
  ) +
  # KL 散度(Scale) 标签放在顶部
  annotate("text", x = 62, y = Inf, label = "KL Divergence\n(Scale)", 
           hjust = 0, family = "Times New Roman", size = 5, fontface = "bold", vjust = 0) +  # vjust 控制顶部位置
  # 在 KL 散度列外部标注 KL 散度
  geom_text(data = data_kl %>% filter(!is.na(normalized_kl)), aes(x = 62, y = habitat, label = round(normalized_kl, 2)),
            inherit.aes = FALSE, hjust = 0, family = "Times New Roman", size = 5, fontface = "bold") +
  coord_cartesian(xlim = c(0, 62), clip = "off")  # 扩展 x 轴范围，允许标注超出图的范围
dev.off() 

