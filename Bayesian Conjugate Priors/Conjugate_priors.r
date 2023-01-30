# Procedemos a graficar diferentes distribuciones
# Definimos una rejilla de valores para nuestra Theta
thetagrid <- seq(0, 5, 0.001)

# Nuestra función a priori sigue una Distribución Gamma
# Asumimos los valores (5, 4) para Alpha y Beta
priori <- dgamma(thetagrid, shape=5, rate=4)
priori <- priori/sum(priori)

# Datos que necesitamos para graficar las densidades
# De una parte el tamaño muestral, y de otra la media de la variable 'Y'
n <- 1 
y <- 4.2 

# Tomamos la expresión de la verosimilitud presentada en el proyecto
vero <- dexp(y, thetagrid)

# Expresamos la función a posteriori resultante
producto <- vero*priori
posteriori <- producto/sum(producto)

# Preparamos nuestro gráfico

fmax <- max(vero, priori, posteriori)
plot(thetagrid, vero, type='l', lwd=2, col='blue', ylim=c(0, fmax), xlab=expression(thetagrid), ylab='f')

# Añadimos la función a priori
lines(thetagrid, priori, type='l', lwd=2, col='red') 

# Y ahora la a posteriori
lines(thetagrid, posteriori, type='l', lwd=2, col= 'green')

# Ajustamos mejor las escalas, los valores y mejoramos la combinación de colores
ylim <- c(0, max(c(priori, posteriori, vero/sum(vero))))
xlim <- range(thetagrid)
plot(thetagrid, priori, type='l', xlim=xlim, ylim=ylim, col='#999999', xlab=expression(theta), ylab='f', yaxt='n')
par(new=T) 
plot(thetagrid, vero/sum(vero), type='l', xlim=xlim, ylim=ylim, col='#E69F00', xlab='', ylab='', yaxt='n')
par(new=T)
plot(thetagrid, posteriori, type='l', xlim=xlim, ylim=ylim, col='#56B4E9', xlab='', ylab='', yaxt='n')
legend('topright', c('Verosimilitud', 'Priori', 'Posteriori'), lty=30, col=c('#999999', '#E69F00', '#56B4E9'), cex=1.2, pch=15)

# Ahora presentamos un caso más realista 
set.seed(42)
theta <- 6
datos <- rexp(120, theta)

# Definimos los parámetros y la rejilla
alpha <- 5
beta <- 4
y_grid <- seq(0, 30, 0.01)

# Cargamos la librería para graficar una distribución de Pareto
library(extraDistr)
plot(dpareto(y_grid, alpha, beta))
plot(dpareto(y_grid, alpha, beta), ylim=c(0.01, 0.8), xlim=c(500, 2500)) # Limitando los valores en la gráfica

# La expresión para el EMV de theta
emv_theta <- 1/mean(datos)

# Verificamos que se aproxima al valor que dimos al principio
# Parámetros de la distribución a posteriori
apost <- alpha + length(datos)
bpost <- beta + length(datos)*mean(datos)
media_post <- apost/bpost

# Terminamos con un gráfico de las funciones
vero <- dgamma(y_grid, length(datos) + 1, sum(datos))
priori <- dgamma(y_grid, alpha, beta)
posteriori <- dgamma(y_grid, apost, bpost)
fmax <- max(vero, priori, posteriori)
plot(y_grid, vero, type='l', lwd=2, col='#999999', ylim=c(0, fmax), xlab=expression(theta), ylab='f')
lines(y_grid, priori, type='l', lwd=2, col='#E69F00')
lines(y_grid, posteriori, type='l', lwd=2, col='#56B4E9')
legend('topright', c('Verosimilitud', 'Priori', 'Posteriori'), lty=30, col=c('#999999', '#E69F00', '#56B4E9'), cex=1.2, pch=15)

# La función predictiva a posteriori viene dada al reemplazar los parámetros de la marginal por los de la a posteriori
rejilla_pred <- seq(0, 5, 0.001)
fy_pred <- dpareto(rejilla_pred + bpost, apost, bpost)
hist(datos ,freq=F, xlab='Datos', ylab='Densidad', main='Distribución Predictiva a posteriori')
lines(rejilla_pred, fy_pred, col='red')

# Es posible combinar varios mediante una mixtura de distribuciones Gamma debidamente ponderadas
# Número de combinaciones para nuestro a priori
k <- 2 
# Pesos inciales de los a prioris según el investigador
w <- c(0.65, 0.35)
# Vector de parámetros alpha
a <- c(5, 6) 
# Vector de parámetros beta
b <- c(4, 3) 
theta <- 6

# Expresamos la mixtura según hemos concluido en el proyecto
set.seed(42)
datos <- rexp(120, theta) 
apost <- a + length(datos)
bpost <- b + length(datos)*mean(datos)
expresion <- ((b^a)*gamma(apost))/((gamma(a)*(bpost)^apost))
wpost <- w*expresion
# Normalizamos los pesos para que sumen a 1
wpost <- wpost/sum(wpost) 

# Graficamos las nuevas distribuciones al introducir la mixtura 
thetagrid <- seq(0, 10, 0.001)
verosimilitud <- dgamma(thetagrid, length(datos) + 1, sum(datos))
fthetapri <- rep(0,1001)
fthetapost <- rep(0,1001)

# Bucle para obtener la distribución a priori y a posteriori 
for (i in 1:k) {
  fthetapri <- fthetapri[i] + w[i]*dgamma(thetagrid, a[i], b[i])
  fthetapost <- fthetapost[i] + wpost[i]*dgamma(thetagrid, apost[i], bpost[i])
}
maxf <- max(c(verosimilitud, fthetapri, fthetapost))
plot(thetagrid, fthetapri, type='l',lwd=2, col='#E69F00', xlab=expression(theta), ylab='f', ylim=c(0, maxf), main='Mixtura de Distribuciones Gamma')
lines(thetagrid, fthetapost,type='l',lwd=2, col='#56B4E9')
lines(thetagrid, verosimilitud, type='l',lwd=2,col='#999999')
legend('topright', c('Verosimilitud', 'Priori', 'Posteriori'), lty=30, col=c('#999999', '#E69F00', '#56B4E9'), cex=1.2, pch=15)

# Se puede aprovechar la mixtura para realizar predicciones
rejilla <- min(datos) + (max(datos) - min(datos))*c(0:1000)/1000
pxpred <- rep(0, length(rejilla) + 1)
for (i in 1:k) {
  pxpred <- pxpred[i] + wpost[i]*dpareto(rejilla + bpost[i], apost[i], bpost[i]) 
}
plot(c(1:length(rejilla)), pxpred, type='l', lwd=2, col='green', xlab='x', ylab='P', main='Distribución Marginal 'Y' con Mixtura')
