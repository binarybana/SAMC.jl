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
  curr :: Float64
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
```

