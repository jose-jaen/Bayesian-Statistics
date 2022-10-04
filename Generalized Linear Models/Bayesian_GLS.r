# Cargamos los datos
rm(list=ls())

df <- read.csv('/home/jose/Desktop/Universidad/5º Curso/Primer cuatrimestre/Métodos Bayesianos/Tercer Proyecto')

# Creamos las dummy variables
df[,'Sex'] <- as.integer(1 * (df$Sex == 'M'))

for (i in 1:dim(df)[1]) {
  if (df$ChestPainType[i] == 'ASY') {
    df$ChestPainType[i] <- 0
  } else if (df$ChestPainType[i] == 'ATA') {
    df$ChestPainType[i] <- 1 
  } else if (df$ChestPainType[i] == 'NAP') {
    df$ChestPainType[i] <- 2
  } else if (df$ChestPainType[i] == 'TA') {
    df$ChestPainType[i] <- 3
  }
}

df[,'ExerciseAngina'] <- as.integer(1 * (df$ExerciseAngina == 'Y'))

for (i in 1:dim(df)[1]) {
  if (df$RestingECG[i] == 'LVH') {
    df$RestingECG[i] <- 0
  } else if (df$RestingECG[i] == 'Normal') {
    df$RestingECG[i] <- 1
  } else if (df$RestingECG[i] == 'ST') {
    df$RestingECG[i] <- 2
  }
}

for (i in 1:dim(df)[1]) {
  if (df$df$ST_Slope[i] == 'Down') {
    df$ST_Slope[i] <- 0
  } else if (df$ST_Slope[i] == 'Flat') {
    df$RestingECG[i] <- 1
  } else if (df$ST_Slope[i] == 'Up') {
    df$ST_Slope[i] <- 2
  }
}

# Aplicamos Regularización L1 mediante Regresión Lasso
library(glmnet)
x <- model.matrix(HeartDisease ~ ., df)
y <- df$HeartDisease

# Observamos la dispersión de 'y' para la rejilla de valores del 'Hyperparameter Tunning'
var(y)
grid <- 10^seq(2, -2, length=100)
fit_lasso <- glmnet(x, y, alpha=1,lambda=grid)
fit_lasso_cv <- cv.glmnet(x, y, alpha=1, lambda=grid, nfolds=nrow(df), grouped=FALSE)
fit_lasso_cv$lambda.min
coef(fit_lasso)[,fit_lasso$lambda == fit_lasso_cv$lambda.min]

# Regresión Logística: Método frecuentista
logit <- glm(HeartDisease ~ (Age + as.factor(Sex) + as.factor(ChestPainType) + Cholesterol + as.factor(FastingBS) + MaxHR + as.factor(ExerciseAngina) + Oldpeak + as.factor(ST_Slope)), data=df, family=binomial)
summary(logit)

# En este modelo se puede observar que las variables Age y MaxHR no son significativas
logit <- glm(HeartDisease ~ ( as.factor(Sex) + as.factor(ChestPainType) + Cholesterol + as.factor(FastingBS) + as.factor(ExerciseAngina) + Oldpeak + as.factor(ST_Slope)), data=df, family=binomial)
summary(logit)

# Predicción con inferencia frecuentista
x <- data.frame(Sex=1, ChestPainType=0, Cholesterol=223, FastingBS=1, ExerciseAngina=1, Oldpeak=0.6, ST_Slope=1)
pred <- predict(logit, x)

# Calculamos la probabilidad
prob <- exp(pred)/(1 + exp(pred))

# Pasamos ahora a la Inferencia Bayesiana
HeartDisease <- df$HeartDisease
Sex <- df$Sex
ChestPainType <- df$ChestPainType
Cholesterol <- df$Cholesterol
FastingBS <- df$FastingBS
ExerciseAngina <- df$ExerciseAngina
Oldpeak <- df$Oldpeak
ST_Slope <- df$ST_Slope
n <- length(Oldpeak)

# Tras definir los predictores preparamos vectores para las variables a utilizar en el muestreo de Gibbs
ChestPainTypeATA <- rep(0, n)
ChestPainTypeNAP <- rep(0, n)
ChestPainTypeTA <- rep(0, n)
ChestPainTypeATA[ChestPainType == '1'] <- 1
ChestPainTypeNAP[ChestPainType == '2'] <- 1
ChestPainTypeTA[ChestPainType == '3'] <- 1

ST_SlopeFlat <- rep(0, n)
ST_SlopeUp <- rep(0, n)
ST_SlopeFlat[ST_Slope == '1'] <- 1
ST_SlopeUp[ST_Slope == '2'] <- 1

# Cargamos los datos 
datos <- list(n=n, HeartDisease=HeartDisease, Sex=Sex, ChestPainTypeATA=ChestPainTypeATA, ChestPainTypeNAP=ChestPainTypeNAP, ChestPainTypeTA=ChestPainTypeTA, Cholesterol=Cholesterol, FastingBS=FastingBS, ExerciseAngina=ExerciseAngina, Oldpeak=Oldpeak, ST_SlopeFlat=ST_SlopeFlat, ST_SlopeUp=ST_SlopeUp)

# Pasamos el código obtenido en Openbugs a R
# Los a prioris no son informativos al carecer de conocimiento suficiente para proponer otros
mlg <- function() {
  for(i in 1:n ) {
    HeartDisease[i] ~ dbern(p[i])
    logit(p[i]) <- beta0 + Cholesterol[i]*betaCholesterol + Sex[i]*betaSex + FastingBS[i]*betaFastingBS + Oldpeak[i]*betaOldpeak + ChestPainTypeATA[i]*betaChestPainTypeATA + ChestPainTypeNAP[i]*betaChestPainTypeNAP + ChestPainTypeTA[i]*betaChestPainTypeTA + ExerciseAngina[i]*betaExerciseAngina + ST_SlopeFlat[i]*betaST_SlopeFlat + ST_SlopeUp[i]*betaST_SlopeUp}
  beta0 ~ dnorm(0.0, 1.0E-6)
  betaCholesterol ~ dnorm(0.0, 1.0E-6)
  betaFastingBS ~ dnorm(0.0, 1.0E-6)
  betaOldpeak ~ dnorm(0.0, 1.0E-6)
  betaExerciseAngina ~ dnorm(0.0, 1.0E-6)
  betaSex ~ dnorm(0.0, 1.0E-6)
  betaChestPainTypeATA ~ dnorm(0.0, 1.0E-6)
  betaChestPainTypeNAP ~ dnorm(0.0, 1.0E-6)
  betaChestPainTypeTA ~ dnorm(0.0, 1.0E-6)
  betaST_SlopeFlat ~ dnorm(0.0, 1.0E-6)
  betaST_SlopeUp ~ dnorm(0.0, 1.0E-6)
}

# Calculamos la probabilidad de poseer una enfermedad cadíaca
p <- length(HeartDisease[HeartDisease == 1])/n

# Calculamos el logit
log(p/(1 - p))

# Cargamos los datos iniciales para las betas
inits <- function() {
  list(beta0=0.214, betaCholesterol=0, betaFastingBS=0, betaOldpeak=0, 
    betaSex=0, betaExerciseAngina=0, betaChestPainTypeATA=0, 
    betaChestPainTypeNAP=0, betaChestPainTypeTA=0, betaST_SlopeFlat=0, 
    betaST_SlopeUp=0)
}

# Obtenemos el modelo mediante métodos MCMC
library(R2OpenBUGS)
mlgout <- bugs(data=datos, inits=inits, parameters.to.save=c('beta0', 'betaSex', 'betaCholesterol', 'betaFastingBS', 'betaOldpeak', 'betaExerciseAngina', 'betaChestPainTypeATA', 'betaChestPainTypeNAP', 'betaChestPainTypeTA', 'betaST_SlopeFlat', 'betaST_SlopeUp'),
               model.file=mlg, n.chains=1, n.burn=10000, n.iter=20000)
mlgout

# Análisis de la convergencia
# Estudiamos la convergencia de $\beta_0$:
beta0 <- mlgout$sims.list$beta0
ts.plot(beta0,xlab='iters',ylab=expression(beta[0]))
ts.plot(cumsum(beta0)/c(1:length(beta0)),xlab='iters',ylab='E[beta0|datos]')
acf(beta0,main='ACF de beta0')

# Estudiamos la convergencia de $\beta_{cholesterol}$
betaCholesterol <- mlgout$sims.list$betaCholesterol
ts.plot(betaCholesterol,xlab='iters',ylab=expression(beta[Cholesterol]))
ts.plot(cumsum(betaCholesterol)/c(1:length(betaCholesterol)),xlab='iters',ylab='E[betaCholesterol|datos]')
acf(betaCholesterol,main='ACF de betaCholesterol')

# Estudiamos la Convergencia de $\beta_{betaFastingBS}$
betaFastingBS <- mlgout$sims.list$betaFastingBS
ts.plot(betaFastingBS,xlab='iters',ylab=expression(beta[FastingBS]))
ts.plot(cumsum(betaFastingBS)/c(1:length(betaFastingBS)),xlab='iters',ylab='E[betaFastingBS|datos]')
acf(betaFastingBS,main='ACF de betaFastingBS')

# Estudiamos la Convergencia de $\beta_{Oldpeak}$
betaOldpeak <- mlgout$sims.list$betaOldpeak
ts.plot(betaOldpeak,xlab='iters',ylab=expression(beta[Oldpeak]))
ts.plot(cumsum(betaOldpeak)/c(1:length(betaOldpeak)),xlab='iters',ylab='E[betaOldpeak|datos]')
acf(betaOldpeak,main='ACF de betaOldpeak')

# Estudiamos la Convergencia de $\beta_{ExerciseAngina}$
betaExerciseAngina <- mlgout$sims.list$betaExerciseAngina
ts.plot(betaExerciseAngina,xlab='iters',ylab=expression(beta[ExerciseAngina]))
ts.plot(cumsum(betaExerciseAngina)/c(1:length(betaExerciseAngina)),xlab='iters',ylab='E[betaExerciseAngina|datos]')
acf(betaExerciseAngina,main='ACF de betaExerciseAngina')

# Estudiamos la Convergencia de $\beta_{Sex}$
betaSex <- mlgout$sims.list$betaSex
ts.plot(betaSex,xlab='iters',ylab=expression(beta[Sex]))
ts.plot(cumsum(betaSex)/c(1:length(betaSex)),xlab='iters',ylab='E[betaSex|datos]')
acf(betaSex,main='ACF de betaSex')

# Estudiamos la Convergencia de $\beta_{ChestPainTypeATA}$
betaChestPainTypeATA <- mlgout$sims.list$betaChestPainTypeATA
ts.plot(betaChestPainTypeATA,xlab='iters',ylab=expression(beta[ChestPainTypeATA]))
ts.plot(cumsum(betaChestPainTypeATA)/c(1:length(betaChestPainTypeATA)),xlab='iters',ylab='E[betaChestPainTypeATA|datos]')
acf(betaChestPainTypeATA,main='ACF de betaChestPainTypeATA')

# Estudiamos la Convergencia de $\beta_{ChestPainTypeNAP}$
betaChestPainTypeNAP <- mlgout$sims.list$betaChestPainTypeNAP
ts.plot(betaChestPainTypeNAP,xlab='iters',ylab=expression(beta[ChestPainTypeNAP]))
ts.plot(cumsum(betaChestPainTypeNAP)/c(1:length(betaChestPainTypeNAP)),xlab='iters',ylab='E[betaChestPainTypeNAP|datos]')
acf(betaChestPainTypeNAP,main='ACF de betaChestPainTypeNAP')

# Estudiamos la Convergencia de $\beta_{ChestPainTypeTA}$
betaChestPainTypeTA <- mlgout$sims.list$betaChestPainTypeTA
ts.plot(betaChestPainTypeTA,xlab='iters',ylab=expression(beta[ChestPainTypeTA]))
ts.plot(cumsum(betaChestPainTypeTA)/c(1:length(betaChestPainTypeTA)),xlab='iters',ylab='E[betaChestPainTypeTA|datos]')
acf(betaChestPainTypeTA,main='ACF de betaChestPainTypeTA')

# Estudiamos la Convergencia de $\beta_{SlopeFlat}$
betaST_SlopeFlat <- mlgout$sims.list$betaST_SlopeFlat
ts.plot(betaST_SlopeFlat,xlab='iters',ylab=expression(beta[ST_SlopeFlat]))
ts.plot(cumsum(betaST_SlopeFlat)/c(1:length(betaST_SlopeFlat)),xlab='iters',ylab='E[betaST_SlopeFlat|datos]')
acf(betaST_SlopeFlat,main='ACF de betaST_SlopeFlat')

# Estudiamos la Convergencia de $\beta_{SlopeUp}$
betaST_SlopeUp <- mlgout$sims.list$betaST_SlopeU
ts.plot(betaST_SlopeUp,xlab='iters',ylab=expression(beta[ST_SlopeUp]))
ts.plot(cumsum(betaST_SlopeUp)/c(1:length(betaST_SlopeUp)),xlab='iters',ylab='E[betaST_SlopeUp|datos]')
acf(betaST_SlopeUp,main='ACF de betaST_SlopeUp')

# Densidades a posteriori
# Estudiamos la densidad a posteriori de $\beta_{0}$
plot(density(beta0),xlab=expression(beta0),ylab='f',main='Densidad a posteriori de beta0')
mean(beta0)
quantile(beta0,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{cholesterol}$
plot(density(betaCholesterol),xlab=expression(betaCholesterol),ylab='f',main='Densidad a posteriori de betacholesterol')
mean(betaCholesterol)
quantile(betaCholesterol,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{betaFastingBS}$
plot(density(betaFastingBS),xlab=expression(betaFastingBS),ylab='f',main='Densidad a posteriori de betaFastingBS')
mean(betaFastingBS)
quantile(betaFastingBS,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{Oldpeak}$
plot(density(betaOldpeak),xlab=expression(betaOldpeak),ylab='f',main='Densidad a posteriori de betaOldpeak')
mean(betaOldpeak)
quantile(betaOldpeak,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{ExerciseAngina}$
plot(density(betaExerciseAngina),xlab=expression(betaExerciseAngina),ylab='f',main='Densidad a posteriori de betaExerciseAngina')
mean(betaExerciseAngina)
quantile(betaExerciseAngina,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{Sex}$
plot(density(betaSex),xlab=expression(betaSex),ylab='f',main='Densidad a posteriori de betaSex')
mean(betaSex)
quantile(betaSex,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{ChestPainTypeATA}$.
plot(density(betaChestPainTypeATA),xlab=expression(betaChestPainTypeATA),ylab='f',main='Densidad a posteriori de betaChestPainTypeATA')
mean(betaChestPainTypeATA)
quantile(betaChestPainTypeATA,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{ChestPainTypeNAP}$
plot(density(betaChestPainTypeNAP),xlab=expression(betaChestPainTypeNAP),ylab='f',main='Densidad a posteriori de betaChestPainTypeNAP')
mean(betaChestPainTypeNAP)
quantile(betaChestPainTypeNAP,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{ChestPainTypeTA}$
plot(density(betaChestPainTypeTA),xlab=expression(betaChestPainTypeTA),ylab='f',main='Densidad a posteriori de betaChestPainTypeTA')
mean(betaChestPainTypeTA)
quantile(betaChestPainTypeTA,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{ST_SlopeFlat}$
plot(density(betaST_SlopeFlat),xlab=expression(betaST_SlopeFlat),ylab='f',main='Densidad a posteriori de betaST_SlopeFlat')
mean(betaST_SlopeFlat)
quantile(betaST_SlopeFlat,c(0.025,0.5,0.975))

# Estudiamos la densidad a posteriori de $\beta_{ST_SlopeUp}$
plot(density(betaST_SlopeUp),xlab=expression(betaST_SlopeUp),ylab='f',main='Densidad a posteriori de betaST_SlopeUp')
mean(betaST_SlopeUp)
quantile(betaST_SlopeUp,c(0.025,0.5,0.975))

# Predicción con inferencia bayesiana
mupred <- beta0 + betaSex*1 + betaCholesterol*223 + betaOldpeak*0.6 + betaFastingBS*1 + betaExerciseAngina*1 + betaST_SlopeFlat*1
pHeartDisease <- exp(mupred)/(1+exp(mupred))
plot(density(pHeartDisease), xlab='p',ylab='f', main='Probabilidad de padecer un problema cardiaco')
mean(pHeartDisease)
quantile(pHeartDisease,c(0.025, 0.5, 0.975))