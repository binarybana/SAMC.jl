module SAMC

export MCMC, Sampler, MHRecord, AMWGRecord, SAMCRecord, set_energy_limits, sample, plotsamc

abstract MCMC
abstract Sampler

#import Distributions.sample
import Base: length
length(x::MCMC) = length(x.db)

# To be overwritten
energy(x::Sampler) = error("Must be instantiated for your sampler object")
propose!(x::Sampler) = error("Must be instantiated for your sampler object")
reject!(x::Sampler) = error("Must be instantiated for your sampler object")
save!(x::Sampler) = error("Must be instantiated for your sampler object")

include("mh.jl")
include("amwg.jl")
include("samc.jl")

include("samc_utils.jl")

end # module
