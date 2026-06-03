# Model parameters 

e=1.527139 #estimated CO2 emissions to atmosphere each year (Gt)
sc= 1.196439 #Standard deviation of Delta c
c=-3.4139122
theta=0.0048565
sx=0.1425355# sd(epsilon')
sy=sx
c2=-1.9572312#
theta2=0.0027880



##########################    ensemble simulations ################################

m=50	         # number of ensemble members
N=500              # maximum length of each ensemble member
k=20 # 
n=seq(1,N)
sim <- function(N,m) # make m time-series each of length N
{
   xt   <- array(0,dim=c(N,m))       # initialise array of x time-series
   yt   <- array(0,dim=c(N,m))       # initialise array of y time-series
   ct   <- array(0,dim=c(N,m))       # initialise array of c time-series
   
   
   
   for (j in 1:m)               # 
   {
      ec=rnorm(N,sd=sc)#
      ct[,j]=630+cumsum(e+ec)# simulate CO2
      
      ey=rnorm(N,sd=sy)# error in y
      yt[,j]=c + theta*ct[,j]+ey		#  "observed"  temperature 
      
      ex=rnorm(N,sd=sx) # error in x
      xt[,j]=c2 + theta2*ct[,j]+ex		     #  "model" temperature
      
      
   }
   list(xt=xt,yt=yt,ct=ct)
   
}




mysim <- sim(N=N,m=m) # get an ensemble simulation

