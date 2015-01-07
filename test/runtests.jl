using Base.Test
using Distributions

using SAMC
import SAMC: energy, propose!, reject!, save!, record

type MySampler <: SAMC.Sampler
  curr :: Float64
  old :: Float64
  data :: Vector{Float64}
end

function propose!(obj::MySampler, block=1, sigma=1.0)
  # block and sigma are for AMWG block updating
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

function energy(obj::MySampler, block=1) # block is for AMWG
  -logpdf(Normal(obj.curr,1.0), obj.data) |> sum
end

########################
# MH
########################
data = rand(5)
#This ^^ initializes the starting point of the chain and the data

mh = MHRecord(MySampler(0., 0., data) ,burn=100)
sample!(mh,2000)

# Now mh.db will be a Vector{Any} of Floats (as in general the Sampler.curr
# type is a pointer type

# posterior average (expectation):
@show posterior_e(identity, mh)
########################
# AMWG
########################
numblocks = 1
amwg = AMWGRecord(MySampler(0., 0., data), numblocks,burn=100)
sample!(amwg,2000)

@show posterior_e(identity, amwg)
########################
# SAMC
########################

samcsamp = set_energy_limits(MySampler(0.,0.,data))
# ^^ Convenience function to set energy limits
samcsamp.stepscale = 10
sample!(samcsamp, 2000)

@show posterior_e(identity, samcsamp)
########################
# POP SAMC
########################
num_chains = 5
genfunc = _ -> MySampler(0., 0., data)

popsamcsamp = set_energy_limits(genfunc, num_chains)
# ^^ Convenience function to set energy limits
popsamcsamp.stepscale = 10
sample!(popsamcsamp, 2000)
# Now popsamcsamp.dbs will be a Vector{Vector{Any}} with all recorded samples
#
@show posterior_e(identity, popsamcsamp)
