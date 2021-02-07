
##################
require(boot)
require(MASS)
require(faraway)
require(caret)
require(leaps)
require(glmnet)
require(origami)
require(stringr)
require(rpart)
require(randomForest)
require(dplyr)
require(gbm)
###################

concrete = read.csv( "~/Documents/CHE course work /CHE spring'20/ST 516/Assignment /Project's /mid term project /concrete.csv")
attach(concrete)
cor(concrete)
pairs(concrete)
plot(age,str)

###################
# centering the data 
#centering...
cement=cement-mean(cement)
slag=slag-mean(slag)
fly_ash=fly_ash-mean(fly_ash)
water=water-mean(water)
superplasticizer=superplasticizer-mean(superplasticizer)
course_aggregate=course_aggregate-mean(course_aggregate)
fine_aggregate=fine_aggregate-mean(fine_aggregate)
age=age-mean(age)

# defining the centered data set 
concrete_c = data.frame(cement,slag,fly_ash,water,superplasticizer,course_aggregate,fine_aggregate,age,strength)
colnames(concrete_c)=c("cement","slag","fly_ash","water","superplasticizer","course_aggregate","fine_aggregate","age","strength")

attach(concrete_c)
#################
#   LINEAR MODEL 
#################
#fitting the linerar model initially 
mod=lm(strength~.,concrete_c)
par(mfrow=c(2,2))
plot(mod)
summary(mod)
vif(mod)
lmod = lm(strength~.,data = train)
new_data= predict(lm(strength~.,train),test, se.fit = TRUE)
Mse_full = mean((test.1$strength-new_data$fit)^2)
Mse_full

 # randomly spliting the samlpe data to test and prediction 
smp_size <- floor(0.75 * nrow(concrete_c)) ## 75% of the sample size
## set the seed to randomize
set.seed(10)
train_ip <- sample(seq_len(nrow(concrete_c)), size = smp_size)
train = concrete_c[train_ip, ]   # defining the train data for prediction 
test  = concrete_c[-train_ip, ]  # defining the test  data for prediction 

##                 VALIDATION
################### 
# fitting a linear model with train data 
lmod = lm(strength~.-couse_aggregate-fine_aggregate,data = train)
par(mfrow=c(2,2))
plot(lmod)
summary(lmod)
vif(lmod)

###

# we find that course aggregate & fine aggregate is not a predictor thta would not contribute to the model. so removing it 
concrete_c.1 = concrete_c[,-(6:7)] 
train.1 = train[,-(6:7)]
test.1 = test[,-(6:7)]

cor(concrete_c.1)

# fitting the model with the required predictors 

lmod.1 = lm(strength~.,data = train.1)
par(mfrow=c(2,2))
plot(lmod.1)
summary(lmod.1)
vif(lmod.1)

# predicting data 
new_data= predict(lm(strength~.,train.1),test, se.fit = TRUE)

## recording MSE errors for models (does k- fold define a model ??)
MSE_l=data.frame(matrix(NA,nrow=6,ncol=4))  # create dummy data frame to store MSE values
colnames(MSE_l)=c("model","linear error","linear interac","poly interac")
MSE_l$model=c("val_linear","LOOCV","K-fold","ridge","LASSO","subset")

## MSE error linear model 
MSE_l[1,2] = mean((test.1$strength-new_data$fit)^2)
MSE_l[1,2]

#######################
#       CROSS VALIDATION- PREDICTABILITY
########################
###            LOOCV 
mod_LOOCV = glm(strength~.,data=train.1) #modeing as linear model 
plot(mod_LOOCV)
summary(mod_LOOCV)


pcv.1 = predict(mod_LOOCV, test.1)
errorcv.1 = (pcv.1- test.1$strength)
MSE_l[2,2] =  mean(errorcv.1^2)
MSE_l[2,2]

######       K- fold

folds=5  # set number of folds
# Fit additive model and estimate test MSE using 5-Fold CV
lmod.cv=train(strength~.,data=train.1,method="lm", 
              trControl=trainControl(method="cv",number=folds))
lmod.cv$results[2]
summary(lmod.cv)
pcv.2 = predict(lmod.cv, test.1)
errorcv.2 = (pcv.2- test.1$strength)
MSE_l[3,2]=mean(errorcv.2^2)  ### define this wtih a proper name 
MSE_l[3,2]
##############################
##### FOR INTERPRETABILITY
##############################

seeds=10 # set seed
#  create predictor matric and vector for response
x = model.matrix(strength~.,train.1)[,-1]
y = train.1$strength
x1 = model.matrix(strength~.,test.1)[,-1]
y1 = test.1$strength
######  Ridge regression
#           change the lamda code 
# create grid for lambda, fit model using all lambdas
grid=10^seq(-5,4,length=300) 
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)  

# plot coefficent values as we change lambda
plot(ridge.mod,xlab="L2 Norm")  # x-axis is in terms of sum(beta^2)
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.ridge=cv.glmnet(x,y,alpha=0,lambda=grid)
plot(cv.ridge)
# finding best lamda
bestlam.r_reg=cv.ridge$lambda.min
pcv.3 = predict(ridge.mod, s=bestlam.r_reg,newx = x1 )
errorcv.3 = (pcv.3- test.1$strength)
MSE_l[4,2]=mean((errorcv.3)^2)
bestlam.r_reg
MSE_l[4,2]

sst <- sum((y1 - mean(y1))^2)
sse <- sum((pcv.3 - y1)^2)
# R squared
rsq <- 1 - sse / sst
rsq
#

########## LASSO
grid=10^seq(-5,4,length=300) 
lasso.mod=glmnet(x,y,alpha=1,lambda=grid)  

# plot coefficent values as we change lambda
plot(lasso.mod,xlab="L2 Norm")  # x-axis is in terms of sum(beta^2)
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.lasso=cv.glmnet(x,y,alpha=1,lambda=grid)
plot(cv.lasso)
bestlam.L_reg=cv.lasso$lambda.min

pcv.4 = predict(lasso.mod, s=bestlam.L_reg,newx = x1 )
errorcv.4 = (pcv.4- test.1$strength)
MSE_l[5,2]=mean((errorcv.4)^2)
bestlam.L_reg
MSE_l[5,2]
summary(lasso.mod)
lasso.coef=predict(cv.lasso,type="coefficients",s=bestlam.L_reg)
lasso.coef

sst <- sum((y1 - mean(y1))^2)
sse <- sum((pcv.4 - y1)^2)
# R squared
rsq <- 1 - sse / sst
rsq

##########################Subset Selection-(K- fold cross validation)                 #

best.mods=regsubsets(strength~.,data=train.1,nvmax=6,method="exhaustive")
summary(best.mods)
a= summary(best.mods)
names(a)
pred.sbs=function(obj,new,id){
  form=as.formula(obj$call[[2]])
  mat=model.matrix(form,new)
  coefi=coef(obj,id=id)
  xvars=names(coefi)
  return(mat[,xvars]%*%coefi)
}
k=5  # set number of folds
set.seed(10)
# create an index with id 1-5 to assign observations to folds
folds=sample(1:k,nrow(train.1),replace=T) 
folds.1= sample(1:k,nrow(test.1),replace=T)
# create dummy matrix to store CV error estimates
cv.err=matrix(NA,k,6,dimnames=list(NULL,paste(1:6)))

# perform CV
for (j in 1:k){
  # pick models with lowest RSS with 1-6 predictors fit without kth fold
  best.mods=regsubsets(strength~.,data=train.1[folds!=j,],
                       nvmax=6,method="exhaustive")
  # estimate test error for all six models by predicting kth fold 
  for (i in 1:6){
    pred=pred.sbs(best.mods,test.1[folds.1==j,],id=i)
    cv.err[j,i]=mean((test.1$strength[folds.1==j]-pred)^2)  # save error est
  }
}

# MSE 
mse.cv_ss=apply(cv.err,2,mean) # compute mean MSE for each number of predictors
min_ss=which.min(mse.cv_ss)  # find minimum mean MSE
#display error 
mse_ss=data.frame(mse.cv_ss)
mse_ss
MSE_l[6,2] = mse_ss$mse.cv_ss[min_ss]
MSE_l[6,2]
reg.summary = summary(best.mods)
reg.summary$rsq


# plot and put a red circle around lowest MSE
  par(mfrow=c(1,1))
plot(1:6,mse.cv_ss,type="b",xlab="no. of predictors)",ylab="est. test MSE",ylim=c(100,220))
points(min_ss,mse.cv_ss[min_ss],cex=2,col="red",lwd=2)
abline(h=c(0,1e-4,2e-4,3e-4,4e-4),lty=3)

# to finalize the model, refit our best subsets p=6 model using the observations

# but we still have some high VIFs so while this model is accurate, perhaps not
# the best for interpretability

# we look for colinearity
##############################
####################################
##   Linear model with interactions 
###################################
#############################
# modeling the interaction 
lmod_i= lm(strength~.+cement:slag+cement:fly_ash+cement:fine_aggregate+slag:fly_ash+slag:course_aggregate
           +slag:fine_aggregate+fly_ash:water+fly_ash:superplasticizer+water:superplasticizer+
             water:fine_aggregate+water:age+superplasticizer:course_aggregate
           +superplasticizer:fine_aggregate,data =train)
par(mfrow=c(2,2))
plot(lmod_i)
summary(lmod_i)
vif(lmod_i)
new_data.1= predict(lmod_i,test, se.fit = TRUE)
mse =mean((test$strength-new_data.1$fit)^2)
mse

# removing the terms that are not contributing _

lmod_i.re= lm(strength~.+cement:fly_ash+slag:course_aggregate
              +fly_ash:superplasticizer+water:superplasticizer+
                water:age+superplasticizer:course_aggregate
              +superplasticizer:fine_aggregate-fine_aggregate-course_aggregate,data =train)
summary(lmod_i.re)

 
#  removing 
# -fine_aggregate-course_aggregate+cement:fine_aggregate+cement:slag+slag:fine_aggregate+water:fine_aggregate+fly_ash:water
# +slag:fly_ash+

par(mfrow=c(2,2))
plot(lmod_i.re)
vif(lmod_i.re)

# refitting the model with train_i data set 

lmod_i.re= lm(strength~.+cement:fly_ash+slag:course_aggregate
              +fly_ash:superplasticizer+water:superplasticizer+
                water:age+superplasticizer:course_aggregate
              +superplasticizer:fine_aggregate-fine_aggregate-course_aggregate,data =train)
summary(lmod_i.re)

par(mfrow=c(2,2))
plot(lmod_i.re)
vif(lmod_i.re)

# predicting data 
new_data.1= predict(lmod_i.re,test, se.fit = TRUE)

## MSE error linear model 
MSE_l[1,3] = mean((test$strength-new_data.1$fit)^2)
MSE_l[1,3]

# defining the new concrete data frame 
new_concrete=data.frame(cbind(cement,slag,fly_ash,water,superplasticizer,age,strength,+cement*fly_ash,slag*course_aggregate
                              ,fly_ash*superplasticizer,water*superplasticizer,
                              water*age,superplasticizer*course_aggregate
                              ,superplasticizer*fine_aggregate))

colnames(new_concrete)=c("cement","slag","fly_ash","water","superplasticizer","age","strength","+cement:fly_ash","slag:course_aggregate"
                         ,"fly_ash:superplasticizer","water:superplasticizer"
                         ,"water:age","superplasticizer:course_aggregate","superplasticizer:fine_aggregate")

# randomly spliting the samlpe data to test and prediction 
smp_size = floor(0.75 * nrow(new_concrete)) ## 75% of the sample size
## set the seed to randomize
set.seed(10)
train_ip = sample(seq_len(nrow(new_concrete)), size = smp_size)
train.2 = new_concrete[train_ip, ]   # defining the train data for prediction 
test.2  = new_concrete[-train_ip, ]  # defining the test  data for prediction 

#######################
#       CROSS VALIDATION- PREDICTABILITY
########################
###            LOOCV 
mod_LOOCV.i = glm(strength~.,data = train.2) #modeing as linear model 
par(mfrow=c(2,2))
plot(mod_LOOCV.i)
summary(mod_LOOCV.i)

pcv.i.1 = predict(mod_LOOCV.i, test.2)
errorcv.i.1 = (pcv.i.1- test.2$strength)
MSE_l[2,3] = mean(errorcv.i.1^2)
MSE_l[2,3]

######       K- fold

folds=5  # set number of folds
# Fit additive model and estimate test MSE using 5-Fold CV
lmod.cv.i=train(strength~.,data = train.2,method="lm", 
              trControl=trainControl(method="cv",number=folds))
lmod.cv.i$results[2]
summary(lmod.cv.i)
plot(lmod.cv.i)
# predicting with test data to find MSE 
pcv.i.2 = predict(lmod.cv.i, test.2)
errorcv.i.2 = (pcv.i.2- test.2$strength)
MSE_l[3,3] = mean(errorcv.i.2^2)
MSE_l[3,3]

##############################
##### FOR INTERPRETABILITY
##############################

 # defining the x & y
seeds=10 # set seed
#  create predictor matric and vector for response
x.1 = model.matrix(strength~.,train.2)[,-1]
y.1 = train.2$strength
x1.1 = model.matrix(strength~.,test.2)[,-1]
y1 = test.2$strength
######  Ridge regression

#           change the lamda code 
# create grid for lambda, fit model using all lambdas
grid=10^seq(-5,4,length=300) 
ridge.mod.i=glmnet(x.1,y.1,alpha=0,lambda=grid)  

# plot coefficent values as we change lambda
plot(ridge.mod.i,xlab="L2 Norm")  # x-axis is in terms of sum(beta^2)
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.ridge.i=cv.glmnet(x.1,y.1,alpha=0,lambda=grid)
plot(cv.ridge.i)
# finding best lamda
bestlam.r_reg.i=cv.ridge.i$lambda.min

# predicting with test data to find MSE 
pcv.i.3  = predict(ridge.mod.i, s=bestlam.r_reg.i,newx = x1.1 )
errorcv.i.3  = (pcv.i.3 - test.2$strength)
MSE_l[4,3]=mean((errorcv.i.3)^2)
bestlam.r_reg.i
MSE_l[4,3]

sst <- sum((y1 - mean(y1))^2)
sse <- sum((pcv.i.3 - y1)^2)
# R squared
rsq <- 1 - sse / sst
rsq

########## LASSO

grid=10^seq(-5,4,length=300) 
lasso.mod.i=glmnet(x.1,y.1,alpha=1,lambda=grid)  

# plot coefficent values as we change lambda
plot(lasso.mod.i,xlab="L2 Norm")  # x-axis is in terms of sum(beta^2)
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.lasso.i=cv.glmnet(x.1,y.1,alpha=1,lambda=grid)
plot(cv.lasso.i)
# finding best lamda
bestlam.L_reg.i=cv.lasso.i$lambda.min
MSE_l[5,3]=min(cv.lasso.i$cvm)
bestlam.L_reg.i
# predicting with test data to find MSE 
pcv.i.4  = predict(lasso.mod.i, s=bestlam.L_reg.i,newx = x1.1 )
errorcv.i.4  = (pcv.i.4 - test.2$strength)
MSE_l[5,3]=mean((errorcv.i.4)^2)
MSE_l[5,3]
summary(lasso.mod.i)
lasso.coef.i=predict(cv.lasso.i,type="coefficients",s=bestlam.L_reg.i)[1:14,]
lasso.coef.i

sst <- sum((y1 - mean(y1))^2)
sse <- sum((pcv.i.4 - y1)^2)
# R squared
rsq <- 1 - sse / sst
rsq

###########
#Subset sclection (k-fold)
##########
# best subset selection
# find the best subset of predictors

best.mods=regsubsets(strength~.,data= train.2,nvmax=13,method="exhaustive")
best.sum=summary(best.mods)
best.sum


pred.sbs=function(obj,new,id){
  form=as.formula(obj$call[[2]])
  mat=model.matrix(form,new)
  coefi=coef(obj,id=id)
  xvars=names(coefi)
  return(mat[,xvars]%*%coefi)
}
k=5  # set number of folds
set.seed(10)
# create an index with id 1-5 to assign observations to folds
folds=sample(1:k,nrow(train.2),replace=T) 
folds.1=sample(1:k,nrow(test.2),replace=T) 
# create dummy matrix to store CV error estimates
cv.err=matrix(NA,k,13,dimnames=list(NULL,paste(1:13)))

# perform CV
for (j in 1:k){
  # pick models with lowest RSS with 1-9 predictors fit without kth fold
  best.mods=regsubsets(strength~.,data=train.2[folds!=j,],
                       nvmax=13,method="exhaustive")
  # estimate test error for all nine models by predicting kth fold 
  for (i in 1:13){
    pred=pred.sbs(best.mods,test.2[folds.1==j,],id=i)
    cv.err[j,i]=mean((test.2$strength[folds.1==j]-pred)^2)  # save error est
  }
}

# MSE 
mse.cv_ss.i=apply(cv.err,2,mean) # compute mean MSE for each number of predictors
min_ss.i=which.min(mse.cv_ss.i)  # find minimum mean MSE
#display error 
mse_ss.i=data.frame(mse.cv_ss.i)
mse_ss.i

MSE_l[6,3] = mse_ss.i$mse.cv_ss.i[min_ss.i]
MSE_l[6,3]

# plot and put a red circle around lowest MSE
par(mfrow=c(1,1))
plot(1:13,mse.cv_ss.i,type="b",xlab="no. of predictors)",ylab="est. test MSE",ylim=c(90,230))
points(min_ss.i,mse.cv_ss.i[min_ss.i],cex=2,col="red",lwd=2)
abline(h=c(0,1e-4,2e-4,3e-4,4e-4),lty=3)

reg.summary = summary(best.mods)
reg.summary$rsq


#  looking at corelation plots
# age, slag & cement might quadratic relation


####################################
##   second order model with interactions 
###################################

lmod_p= lm(strength~.+cement:slag+cement:fly_ash+cement:fine_aggregate+slag:fly_ash+slag:course_aggregate
           +slag:fine_aggregate+fly_ash:superplasticizer+water:superplasticizer+fly_ash:water
           +water:fine_aggregate+water:age+superplasticizer:course_aggregate
           +superplasticizer:fine_aggregate+I(age^2)+I(slag^2)+I(cement^2)
           +I(superplasticizer^2)+I(water^2),data =train)


par(mfrow=c(2,2))
plot(lmod_p)
summary(lmod_p)
vif(lmod_p)

new_data.2= predict(lm(lmod_p.re,train),test, se.fit = TRUE)
MSe = mean((test$strength-new_data.2$fit)^2)
MSe
# removing the terms that are not contributing _
lmod_p.re= lm(strength~.+cement:slag+slag:course_aggregate
              +fly_ash:superplasticizer+water:superplasticizer
              +water:age+superplasticizer:course_aggregate
              +superplasticizer:fine_aggregate+I(age^2)+I(slag^2)+I(cement^2)
              -course_aggregate-fine_aggregate,data = train)


#  fly_ash:water+cement:fine_aggregate+cement:fly_ash+I(superplasticizer^2)+slag:fine_aggregate+water:fine_aggregate
# -course_aggregate- fine_aggregate +I(water^2)+slag:fly_ash

par(mfrow=c(2,2))
plot(lmod_p.re)
summary(lmod_p.re)
vif(lmod_p.re)

# predicting this model with test data set 
new_data.2= predict(lm(lmod_p.re,train),test, se.fit = TRUE)

## MSE error linear model 
MSE_l[1,4] = mean((test$strength-new_data.2$fit)^2)
MSE_l[1,4]

###### defining a new data frame that contains all the predictors 

new_concrete.2=data.frame(cbind(cement,slag,fly_ash,water,superplasticizer,age,strength,cement*slag,slag*course_aggregate
                                ,fly_ash*superplasticizer,water*superplasticizer
                                ,water*age,superplasticizer*course_aggregate
                                ,superplasticizer*fine_aggregate,age^2,slag^2,cement^2))
colnames(new_concrete.2)=c("cement","slag","fly_ash","water","superplasticizer","age","strength","cement:slag","slag:course_aggregate","fly_ash:superplasticizer"
                           ,"water:superplasticizer","water:age","superplasticizer:course_aggregate"
                           ,"superplasticizer:fine_aggregate","I(age^2)","I(slag^2)","I(cement^2)")

# randomly spliting the samlpe data to test and prediction 
smp_size = floor(0.75 * nrow(new_concrete.2)) ## 75% of the sample size
## set the seed to randomize
set.seed(10)
train_ip = sample(seq_len(nrow(new_concrete.2)), size = smp_size)
train.3 = new_concrete.2[train_ip, ]   # defining the train data for prediction 
test.3  = new_concrete.2[-train_ip, ]  # defining the test  data for prediction 

#######################
#       CROSS VALIDATION- PREDICTABILITY
########################
###            LOOCV 
mod_LOOCV.pi = glm(strength~.,data = train.3) #modeing as polynomial model 
plot(mod_LOOCV.pi)
summary(mod_LOOCV.pi)
# predicting the model with test data and finding MSE 
pcv.pi.1 = predict(mod_LOOCV.pi, test.3)
errorcv.pi.1 = (pcv.pi.1- test.3$strength)
MSE_l[2,4] = mean(errorcv.pi.1^2)
MSE_l[2,4]


######       K- fold

folds=5  # set number of folds
# Fit additive model and estimate test MSE using 5-Fold CV
lmod.cv.pi=train(strength~.,data = train.3,method="lm", 
                trControl=trainControl(method="cv",number=folds))
lmod.cv.pi$results[2]

# predicting the model with test data and finding MSE 
pcv.pi.2 = predict(lmod.cv.pi, test.3)
errorcv.pi.2 = (pcv.pi.2- test.3$strength)
MSE_l[3,4] = mean(errorcv.pi.2^2)
MSE_l[3,4]
summary(lmod.cv.pi)

##############################
##### FOR INTERPRETABILITY
##############################

# defining the x & y
seeds=10 # set seed
#  create predictor matric and vector for response
x.2 = model.matrix(strength~.,train.3)[,-1]
y.2 = train.3$strength
x1.2 = model.matrix(strength~.,test.3)[,-1]
y1 = test.3$strength
######  Ridge regression

#           change the lamda code 
# create grid for lambda, fit model using all lambdas
grid=10^seq(-5,4,length=300) 
ridge.mod.p=glmnet(x.2,y.2,alpha=0,lambda=grid)  

# plot coefficent values as we change lambda
plot(ridge.mod.p,xlab="L2 Norm")  # x-axis is in terms of sum(beta^2)
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.ridge.p=cv.glmnet(x.2,y.2,alpha=0,lambda=grid)
plot(cv.ridge.p)
# finding best lamda
bestlam.r_reg.p=cv.ridge.p$lambda.min
# predicting the model with test data and finding MSE 
pcv.pi.3  = predict(ridge.mod.p, s=bestlam.r_reg.p,newx = x1.2 )
errorcv.pi.3  = (pcv.pi.3 - test.3$strength)
MSE_l[4,4]=mean((errorcv.pi.3)^2)
bestlam.r_reg.p
MSE_l[4,4]

sst <- sum((y1 - mean(y1))^2)
sse <- sum((pcv.pi.3 - y1)^2)
# R squared
rsq <- 1 - sse / sst
rsq


########## LASSO

grid=10^seq(-5,4,length=300) 
lasso.mod.p=glmnet(x.2,y.2,alpha=1,lambda=grid)  

# plot coefficent values as we change lambda
plot(lasso.mod.p,xlab="L2 Norm")  # x-axis is in terms of sum(beta^2)
abline(h=0,lty=3)

# optimize lambda using cross-validation
cv.lasso.p=cv.glmnet(x.2,y.2,alpha=1,lambda=grid)
plot(cv.lasso.p)
bestlam.L_reg.p=cv.lasso.p$lambda.min
# predicting the model with test data and finding MSE 
pcv.pi.4  = predict(lasso.mod.p, s=bestlam.L_reg.p,newx = x1.2 )
errorcv.pi.4  = (pcv.pi.4 - test.3$strength)
MSE_l[5,4]=mean((errorcv.pi.4)^2)
bestlam.L_reg.p
MSE_l[5,4]
summary(lasso.mod.p)
lasso.coef.p=predict(cv.lasso.p,type="coefficients",s=bestlam.L_reg.p)[1:17,]
lasso.coef.p

sst <- sum((y1 - mean(y1))^2)
sse <- sum((pcv.pi.4 - y1)^2)
# R squared
rsq <- 1 - sse / sst
rsq


###########
#Subset sclection (k-fold)
##########
# best subset selection
# find the best subset of predictors

best.mods=regsubsets(strength~.,data=train.3,nvmax=16,method="exhaustive")
best.sum=summary(best.mods)
best.sum$rsq


pred.sbs=function(obj,new,id){
  form=as.formula(obj$call[[2]])
  mat=model.matrix(form,new)
  coefi=coef(obj,id=id)
  xvars=names(coefi)
  return(mat[,xvars]%*%coefi)
}
k=5  # set number of folds
set.seed(10)
# create an index with id 1-5 to assign observations to folds
folds=sample(1:k,nrow(train.3),replace=T) 
folds.1=sample(1:k,nrow(test.3),replace=T) 
# create dummy matrix to store CV error estimates
cv.err=matrix(NA,k,16,dimnames=list(NULL,paste(1:16)))

# perform CV
for (j in 1:k){
  # pick models with lowest RSS with 1-16 predictors fit without kth fold
  best.mods=regsubsets(strength~.,data=train.3[folds!=j,],
                       nvmax=16,method="exhaustive")
  # estimate test error for all 16 models by predicting kth fold 
  for (i in 1:16){
    pred=pred.sbs(best.mods,test.3[folds.1==j,],id=i)
    cv.err[j,i]=mean((test.3$strength[folds.1==j]-pred)^2)  # save error est w.r.t test data 
  }
}

# MSE 
mse.cv_ss.p=apply(cv.err,2,mean) # compute mean MSE for each number of predictors
min_ss.p=which.min(mse.cv_ss.p)  # find minimum mean MSE
#display error 
mse_ss.p=data.frame(mse.cv_ss.p)
mse_ss.p
MSE_l[6,4] = mse_ss.p$mse.cv_ss.p[min_ss.p]
MSE_l[6,4]

# plot and put a red circle around lowest MSE
par(mfrow=c(1,1))
plot(1:16,mse.cv_ss.p,type="b",xlab="no. of predictors)",ylab="est. test MSE",ylim=c(60,210))
points(min_ss.p,mse.cv_ss.p[min_ss.p],cex=2,col="red",lwd=2)
abline(h=c(0,1e-4,2e-4,3e-4,4e-4),lty=3)

reg.summary = summary(best.mods)
reg.summary$rsq


### ploting residual vs fitted models by the best method for all the models 

# actual vs fitted 
# linear model 
plot(new_data$fit,test.1$strength,pch=19,xlab= "predicted", ylab = "actual",xlim =c(0,80),ylim=c(0,80))
abline(0,1,col="blue")

# linear model with interaction
plot(new_data.1$fit,test.2$strength,pch=19,xlab= "predicted", ylab = "actual",xlim =c(0,80),ylim=c(0,80))
abline(0,1,col="blue")

# polynomial model with interactions 
plot(new_data.2$fit,test.3$strength,pch=19,,col="red",xlab= "predicted", ylab = "actual",xlim =c(0,80),ylim=c(0,80))
abline(0,1,col="blue")



