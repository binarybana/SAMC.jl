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
data = [0.0, 0.0, 1.0, 1.0]

mh = MHRecord(MySampler(0., 0., data) ,burn=100)
sample!(mh,20000)

# Now mh.db will be a Vector{Any} of Floats (as in general the Sampler.curr
# type is a pointer type

# posterior average (expectation):
@show posterior_e(identity, mh)
mh_samps = samples(mh) # Get posterior samples
mh_mape = mapenergy(mh) # Get MAP energy
mh_mapv = mapvalue(mh) # Get the MAP energy sample

@test ndims(mh_samps) == 1
@test length(mh_samps) > 0
@test 0.45 < mh_mapv < 0.55
@test 4 < mh_mape < 5

########################
# MH Multiple Chains
########################
mhchains = SAMC.MHRecord[MHRecord(MySampler(0., 0., data), burn=100) for x=1:10]
sample!(mhchains, round(Int, 20000/10))

@show posterior_e(identity, mhchains)
mhc_samps = samples(mhchains) # Get posterior samples
mhc_mape = mapenergy(mhchains) # Get MAP energy
mhc_mapv = mapvalue(mhchains) # Get the MAP energy sample
mhc_gelman = gelman_rubin(identity, mhchains) # Gelman Rubin Statistic 
# (only for multiple chain MH atm)

@test ndims(mhc_samps) == 1
@test length(mhc_samps) > 0
@test 0.45 < mhc_mapv < 0.55
@test 4 < mhc_mape < 5
@test 0.95 < mhc_gelman[1] < 1.05

########################
# AMWG
########################
numblocks = 1
amwg = AMWGRecord(MySampler(0., 0., data), numblocks, burn=100)
sample!(amwg, 20000)

@show posterior_e(identity, amwg)
ag_samps = samples(amwg) # Get posterior samples
ag_mape = mapenergy(amwg) # Get MAP energy
ag_mapv = mapvalue(amwg) # Get the MAP energy sample

@test ndims(ag_samps) == 1
@test length(ag_samps) > 0
@test 0.45 < ag_mapv < 0.55
@test 4 < ag_mape < 5
########################
# SAMC
########################

samcsamp = set_energy_limits(MySampler(0.,0.,data))
# ^^ Convenience function to set energy limits
samcsamp.stepscale = 10
sample!(samcsamp, 20000)

@show posterior_e(identity, samcsamp)
# s_samps = samples(samcsamp) # Resampling not currently implemented with weighted SAMC
s_mape = mapenergy(samcsamp) # Get MAP energy
s_mapv = mapvalue(samcsamp) # Get the MAP energy sample
@test 0.45 < s_mapv < 0.55
@test 4 < s_mape < 5
########################
# POP SAMC
########################
num_chains = 5
genfunc = _ -> MySampler(0., 0., data)

popsamcsamp = set_energy_limits(genfunc, num_chains)
# ^^ Convenience function to set energy limits
popsamcsamp.stepscale = 10
sample!(popsamcsamp, div(20000,10))
# Now popsamcsamp.dbs will be a Vector{Vector{Any}} with all recorded samples
#
@show posterior_e(identity, popsamcsamp)
# ps_samps = samples(popsamcsamp) # Resampling not currently implemented with weighted SAMC
ps_mape = mapenergy(popsamcsamp) # Get MAP energy
ps_mapv = mapvalue(popsamcsamp) # Get the MAP energy sample

@test 0.45 < ps_mapv < 0.55
@test 4 < ps_mape < 5
