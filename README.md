# SAMC

A library to implement Stochastic Approximation Markov Chain Monte Carlo (SAMC) 
in Julia. 

It also includes Metropolis-Hastings and Adaptive Metropolis-within-Gibbs as 
simpler techniques that can be compared using the same API.

## Usage
As a short example for performing MCMC over a univariate parameter:

```julia
using Base.Test
using Distributions

using SAMC
import SAMC: energy, propose!, reject!, save!, record

type MySampler <: SAMC.Sampler
  curr :: Float64 
  old :: Float64
  data :: Vector{Float64}
end

function propose!(obj::MySampler)
  obj.curr += randn()
  return 0 # Return the scheme or block we updated 
end

function reject!(obj::MySampler)
  obj.curr = obj.old
end

function save!(obj::MySampler)
  obj.old = obj.curr
end

function record(obj::MySampler)
  return obj.curr
end

function energy(obj::MySampler)
  logpdf(Normal(obj.curr,1.0), obj.data) |> sum
end

data = rand(5)
sampler = MySampler(0., 0., data) 
#This ^^ initializes the starting point of the chain and the data

mh = MHRecord(sampler,burn=100)
sample!(mh,2000)

# Now mh.db will be a Vector{Any} of Floats (as in general we don't yet test
# the result of record() and try to specialize the container appropriately
```

Now to use SAMC (or more specifically, the population variant), we could do:

```julia
num_chains = 5
genfunc = _ -> MySampler(0., 0., data)

popsamcsamp = set_energy_limits(genfunc, num_chains) 
# ^^ Convenience function to set energy limits
popsamcsamp.stepscale = 10
sample!(popsamcsamp, 2000)

# Now popsamcsamp.dbs will be a Vector{Vector{Any}} with all recorded samples
```

