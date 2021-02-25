# Make a ggplot for YOLO cross-validation 
library(ggplot2)

# Define CV directory
# args = commandArgs(trailingOnly=TRUE)
cv_dir <- "logs/CV/"

# Read validation output
val_result <- read.csv(paste0(cv_dir, 'CV_results.csv'), row.names = 1)

# Compute stats
val_means <- aggregate(val_result$mAP, by=list(val_result$n), FUN=mean)
val_groups <- val_means$Group.1
val_means <- val_means$x
val_CI_low <- aggregate(val_result$mAP, by=list(val_result$n), 
                        FUN=function(x) tryCatch(return(t.test(x)$conf.int[1]), 
                                                 warning = return(t.test(x)$conf.int[1]),
                                                 error = return(0)))$x  
val_CI_high <- aggregate(val_result$mAP, by=list(val_result$n), 
                         FUN=function(x) tryCatch(return(t.test(x)$conf.int[2]), 
                                                  warning = return(t.test(x)$conf.int[2]),
                                                  error = return(0)))$x

# Cut the intervals
val_means <- apply(as.data.frame(val_means), 1, FUN = function(x) ifelse(x < 0, 0, x))
val_CI_low <- apply(as.data.frame(val_CI_low), 1, FUN = function(x) ifelse(x < 0, 0, x))
val_CI_high <- apply(as.data.frame(val_CI_high), 1, FUN = function(x) ifelse(x < 0, 0, x))

# Makes stats dataframe
stats_data <- as.data.frame(val_means)
stats_data <- cbind(stats_data, val_CI_low)
stats_data <- cbind(stats_data, val_CI_high)
stats_data$iteration <- val_groups
rownames(stats_data) <- val_groups

# Make a ggplot
png(paste0(cv_dir, 'CV_plot.png'))
ggplot(stats_data, aes(x = iteration, y = val_means)) +
  geom_errorbar(aes(ymin=val_CI_low, ymax=val_CI_high), width=1) +
  geom_line(size = 1) +
  geom_point() +
  geom_smooth(size = 0.5) + 
  geom_text(aes(label = round(val_means, 2)), col = 'red') +
  geom_text(aes(label = round(val_CI_low, 2)), vjust = 2) +
  geom_text(aes(label = round(val_CI_high, 2)), vjust = -1) +
  ggtitle('Cross-validation mAP (mean average precision)') +
  xlab('Iteration') +
  ylab('CV mAP@0.50') + 
  geom_hline(yintercept=max(val_means), linetype="dashed", color = "blue", size=0.5) +
  geom_hline(yintercept=summary(val_means)[2], linetype="dashed", color = "darkgrey", size=0.5) +
  geom_vline(xintercept=val_result$n[which.max(val_means)], linetype="dashed", color = "blue", size=0.5) +
  scale_x_continuous(breaks = val_result$n, minor_breaks = NULL, limits = c(0, max(val_groups))) + 
  scale_y_continuous(breaks = seq(0, 1, 0.1), minor_breaks = NULL, limits = c(0, 1))
dev.off()
