module SAMC

using StatsBase
import StatsBase: sample!

export MCMC,
       Sampler,
       MHRecord,
       AMWGRecord,
       SAMCRecord,
       PopSAMCRecord,
       set_energy_limits,
       sample!,
       plotsamc,
       posterior_e,
       cum_posterior_e,
       samples,
       mapvalue,
       mapenergy,
       gelman_rubin

abstract MCMC
abstract Sampler

#import Distributions.sample
import Base: length
length(x::MCMC) = length(x.db)

# To be overwritten
energy(x::Sampler) = error("The energy function must be instantiated for your sampler object")
propose!(x::Sampler) = error("The propose! function must be instantiated for your sampler object")
reject!(x::Sampler) = error("The reject! must be instantiated for your sampler object")
save!(x::Sampler) = error("The save! function must be instantiated for your sampler object")
record(x::Sampler) = error("The record function be instantiated for your sampler object")

samples(x::MCMC) = error("The samples function is not defined for this type of MCMC sampler")
mapvalue(x::MCMC) = error("The mapvalue function is not defined for this type of MCMC sampler")
mapenergy(x::MCMC) = error("The mapenergy function is not defined for this type of MCMC sampler")

include("mh.jl")
include("amwg.jl")
include("samc_sampler.jl")
include("popsamc.jl")

# include("samc_utils.jl") # Until I can do a conditional import on PyPlot
#here

end # module
