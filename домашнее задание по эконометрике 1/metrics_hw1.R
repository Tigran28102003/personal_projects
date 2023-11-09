set.seed(80540)
Y <- round(rnorm(100), 2)
X1 <- round(rnorm(100, mean = 5), 2)
X2 <- round(runif(100, min = 5, max = 20), 2)
myData <- data.frame(Y, X1, X2)
write.csv2(myData, "myData.csv", row.names = F)

#Задача 1. Найди минимальное значение Y в твоей выборке
min(Y)

#Задача 2. Найди дисперсию Y в твоей выборке
var(Y)

#Задача 3. Найди среднее значение Y среди наблюдений, где X2 больше 10
mean(myData$Y[myData$X2 > 10])

#Задача 4. Проверь при помощи t-теста гипотезу о том, что матожидание Y равно 0
#В ответ введи p-value
t.test(myData$Y, mu = 0)$p.value

#Задача 5.Найди корреляцию Y и X2
cor(myData$Y, myData$X2)

#Задача 6. Найди коэффициент при X2 в регрессии Y на X1 и X2
model <- lm(formula=Y~X2+X1, data=myData)
sum_model <- summary(model)
sum_model$coefficients[2, 1]

#Задача 7.Найди R^2 в регрессии Y на X1 и X2
sum_model$r.squared

#Задача 8. Найди t-статистику для гипотезы о равенстве коэффициента при X1 нулю 
# в регрессии Y на X1 и X2
t_stat <- coef(sum_model)["X1", "t value"]
t_stat

#Задача 9. Найди оценку дисперсии ошибок в регрессии Y на X1 и X2
residual_variance <- sum_model$sigma^2
residual_variance

#Задача 10. Найди коэффициент при X2 в регрессии Y на X1 и X2 без константы(!)
model_no_const <- lm(Y ~ X1 + X2 - 1, data=myData)
coef_x2 <- coef(model_no_const)['X2']
coef_x2

install.packages("plotly")

# Создайте график рассеяния с линией регрессии
library(plotly)
scatter_plot <- plot_ly(myData, x = ~X1, y = ~Y, type = 'scatter', 
                        mode = 'markers', marker = list(color = 'blue'))
scatter_plot <- add_lines(scatter_plot, x = myData$X1, 
                             y = predict(model), line = list(color = 'red'))
scatter_plot <- layout(scatter_plot, xaxis = list(title = "X1"), 
                       yaxis = list(title = "Y"), 
                       title = "Scatter Plot with Regression Line")
scatter_plot


