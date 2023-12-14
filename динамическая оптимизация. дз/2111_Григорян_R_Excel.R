a1 <- 20
a2 <- 10
a3 <- 4
b1 <- 4
b2 <- 18
b3 <- 10
c1 <- 1
c2 <- 18
c3 <- 20

# importing packages:
#install.packages('GenSA')
library('GenSA') # genetic algorithm for optimization
#install.packages('alabama')
library('alabama')
library(ggplot2)

# problem 1

t0 <- 0 # initial time moment
tT <- 2 # end time moment
dt <- 0.01 # time step
n_time_moments <- (tT - t0) / dt + 1 # number of time moments
t <- seq(from = t0,
         to = tT, 
         by = dt) # time moments

y_t0 <- -b2 # initial value of target function, y(t0)
y_tT <- b1 # end value of target function, y(tT)

# max:

functional <- function(y_optim)
{
  y <- c(y_t0, y_optim, y_tT)
  dy <- (y - c(y_t0, y[-n_time_moments]))
  f <- (dy/dt)^2 + b1*(y)^2 + c1*(dy/dt)*exp(4*t)
  return(sum(f * dt))
} # the value of the target functional

ga_functional <- GenSA(par = rep((y_t0 + y_tT) / 2, n_time_moments - 2),
                       fn = functional,
                       lower = rep(-1000, n_time_moments - 2),
                       upper = rep(1000, n_time_moments - 2)) # genetic algorithm

ga_functional$value # estimated value of the functional
ga_functional$par # estimated values of the target function at each time moment

y_real <- function(t){
  y <- exp(4*t)/6 * (-1) + ((77 + 96*exp(4) + exp(12))*exp(2*t))/(-6 + 6*exp(8)) + 
    ((-77*exp(8) - exp(12) - 96*exp(4))*exp(-2*t))/(-6 + 6*exp(8))
}

ggplot() + 
  geom_point(mapping = aes(x=t, y=c(y_t0, ga_functional$par, y_tT)), 
             color='purple', alpha=.7) +
  labs(title="Estimated y trajectories",
       x='time moments',
       y='y') + theme_bw()

# min:

functional <- function(y_optim)
{
  y <- c(y_t0, y_optim, y_tT)
  dy <- (y - c(y_t0, y[-n_time_moments]))
  f <- (dy/dt)^2 + b1*(y)^2 + c1*(dy/dt)*exp(4*t)
  return(-sum(f * dt))
} # the value of the target functional

ga_functional <- GenSA(par = rep((y_t0 + y_tT) / 2, n_time_moments - 2),
                       fn = functional,
                       lower = rep(-10000000000, n_time_moments - 2),
                       upper = rep(10000000000, n_time_moments - 2)) # genetic algorithm

ga_functional$value # estimated value of the functional
ga_functional$par # estimated values of the target function at each time moment

ggplot() + geom_line(aes(x=t, y=c(y_t0, ga_functional$par, y_tT), color='red')) +
  labs(title="Eestimated y trajectories",
       x="time moments", y="y")

# problem 2

t0 <- 0 # initial time moment
tT <- 3 # end time moment
dt <- 0.1 # time step
n <- (tT - t0) / dt + 1 # number of time moments
t <- seq(from = t0,
         to = tT, 
         by = dt) # time moments

y_t0 <- a1 # initial value of target function, y(t0)

# min:

functional <- function(u)
{
  y <- c(y_t0, rep(0, n - 1))
  dy <- rep(0, n - 1)
  for (i in 2:n)
  {
    dy[i - 1] <- a3*y[i - 1] + u[i - 1]
    y[i] <- y[i - 1] + dy[i - 1] * dt # y(t) = y(t-1) + y'(t-1) * dt <=> y'(t-1) = (y(t) - y(t-1)) / dt
  }
  return(-sum(b1*y[-1] - b2*(u^2)))
} # the value of the target functional

ga_functional <- GenSA(fn = functional,
                       lower = rep(-c1, n - 1),
                       upper = rep(c2, n - 1)) # genetic algorithm

ga_functional$par
ga_functional$value

ggplot(mapping = aes(x=t[-1], y=ga_functional$par)) + 
  geom_line(color='purple') +
  labs(title="Estimated u trajectory",
                                x='time moments',
                                y='u') + theme_bw()

# max:

functional <- function(u)
{
  y <- c(y_t0, rep(0, n - 1))
  dy <- rep(0, n - 1)
  for (i in 2:n)
  {
    dy[i - 1] <- a3*y[i - 1] + u[i - 1]
    y[i] <- y[i - 1] + dy[i - 1] * dt # y(t) = y(t-1) + y'(t-1) * dt <=> y'(t-1) = (y(t) - y(t-1)) / dt
  }
  return(sum(b1*y[-1] - b2*(u^2)))
} # the value of the target functional

ga_functional <- GenSA(fn = functional,
                       lower = rep(-c1, n - 1),
                       upper = rep(c2, n - 1)) # genetic algorithm

ga_functional$par
ga_functional$value

ggplot(mapping = aes(x=t[-1], y=ga_functional$par)) + 
  geom_line(color='purple') +
  labs(title="Estimated u trajectory",
       x='time moments',
       y='u') + theme_bw()

# problem 7

t0 <- 0 # initial time moment
tT <- 2 # end time moment
dt <- 0.1 # time step
n <- (tT - t0) / dt + 1 # number of time moments
t <- seq(from = t0,
         to = tT, 
         by = dt) # time moments

y_t0 <- 0 # initial value of target function, y(t0)

# пункт a:

# min:

functional <- function(u)
{
  y <- c(y_t0, rep(0, n - 1))
  dy <- rep(0, n - 1)
  for (i in 2:n)
  {
    dy[i - 1] <- u[i - 1] - t[i - 1]
    y[i] <- y[i - 1] + dy[i - 1] * dt # y(t) = y(t-1) + y'(t-1) * dt <=> y'(t-1) = (y(t) - y(t-1)) / dt
  }
  return(-sum((u^2)/2-t[-1]*y[-1]+y[-1]))
} # the value of the target functional

ga_functional <- GenSA(fn = functional,
                       lower = rep(-3/8, n - 1),
                       upper = rep(3/8, n - 1)) # genetic algorithm

u_trajectory <- ga_functional$par
optim_functional <- ga_functional$value

y_trajectory <- c(y_t0, rep(0, n - 1))
dy <- rep(0, n - 1)
for (i in 2:n)
{
  dy[i - 1] <- u_trajectory[i - 1] - t[i - 1]
  y_trajectory[i] <- y_trajectory[i - 1] + dy[i - 1] * dt # y(t) = y(t-1) + y'(t-1) * dt <=> y'(t-1) = (y(t) - y(t-1)) / dt
}

ggplot(mapping = aes(x=t[-1], y=ga_functional$par)) + 
  geom_line(color='purple') +
  geom_line(aes(x=t[-1], y=y_trajectory[-1]), color='darkblue') +
  labs(title="Estimated u and y trajectory",
       x='time moments',
       y='u, y') + theme_bw()


# пункт b:

# min:

functional <- function(u)
{
  y <- c(y_t0, rep(0, n - 1))
  dy <- rep(0, n - 1)
  for (i in 2:n)
  {
    dy[i - 1] <- u[i - 1] + (u[i - 1])^2 - t[i-1]
    y[i] <- y[i - 1] + dy[i - 1] * dt # y(t) = y(t-1) + y'(t-1) * dt <=> y'(t-1) = (y(t) - y(t-1)) / dt
  }
  return(-sum((u^2)/2-t[-1]*y[-1]+y[-1]))
} # the value of the target functional

ga_functional <- GenSA(fn = functional,
                       lower = rep(-3/8, n - 1),
                       upper = rep(3/8, n - 1)) # genetic algorithm

u_trajectory <- ga_functional$par
optim_functional <- ga_functional$value

y_trajectory <- c(y_t0, rep(0, n - 1))
dy <- rep(0, n - 1)
for (i in 2:n)
{
  dy[i - 1] <- u_trajectory[i - 1] - t[i - 1]
  y_trajectory[i] <- y_trajectory[i - 1] + dy[i - 1] * dt # y(t) = y(t-1) + y'(t-1) * dt <=> y'(t-1) = (y(t) - y(t-1)) / dt
}

ggplot(mapping = aes(x=t[-1], y=ga_functional$par)) + 
  geom_line(color='maroon') +
  geom_line(aes(x=t[-1], y=y_trajectory[-1]), color='skyblue') +
  labs(title="Estimated u and y trajectory",
       x='time moments',
       y='u, y') + theme_bw()
