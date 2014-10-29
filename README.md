# SAMC

A library to implement Stochastic Approximation Markov Chain Monte Carlo (SAMC) 
in Julia. 

It also includes Metropolis-Hastings and Adaptive Metropolis-within-Gibbs as 
simpler techniques that can be compared using the same API.

## Usage
As a short example for performing MCMC over a univariate parameter:

```julia
using Distributions
using SAMC
import SAMC: energy, propose!, reject!, save!

type MySampler <: SAMC.Sampler
  curr :: Float64 #Currently must be named curr
  old :: Float64
  data :: Matrix{Float64}
end

function propose!(obj::MySampler)
  obj.curr += randn()
end

function reject!(obj::MySampler)
  obj.curr = obj.old
end

function save!(obj::MySampler)
  obj.old = obj.curr
end

function energy(obj::MySampler)
  logpdf(Normal(obj.curr,1.0), data) |> sum
end

sampler = MySampler(0.,0.) #This initializes the starting point of the chain
mh = MHRecord(sampler)
sample!(mh,2000,burn=1000)

# Now mh.db will be a Vector{Any} of Floats (as in general the Sampler.curr
# type is a pointer type

```

