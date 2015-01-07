using Base.Test
using Distributions

using SAMC
import SAMC: energy, propose!, reject!, save!

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

function energy(obj::MySampler)
  logpdf(Normal(obj.curr,1.0), obj.data) |> sum
end

sampler = MySampler(0., 0., rand(5)) 
#This ^^ initializes the starting point of the chain and the data

mh = MHRecord(sampler,burn=100)
sample!(mh,2000)

# Now mh.db will be a Vector{Any} of Floats (as in general the Sampler.curr
# type is a pointer type
