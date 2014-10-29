module SAMC

export MCMC, Sampler
export MHRecord, AMWGRecord, SAMCRecord, set_energy_limits, sample

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

end # module
