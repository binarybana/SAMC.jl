using Base.Test
using Distributions
using StatsBase

using SAMC
import SAMC: energy, propose!, reject!, save!, record

type MySampler <: SAMC.Sampler
  curr :: Int
  old :: Int
  data :: Vector{Float64}
end

function propose!(obj::MySampler, block=1, sigma=1.0)
  # block and sigma are for AMWG block updating
  # obj.curr += randn()
  obj.curr = rand(1:10)
  # obj.curr = sample(samp_freqs[obj.curr])
  # if randbool()
  #   obj.curr += 1
  # else
  #   obj.curr -= 1
  # end
  # obj.curr = mod(obj.curr-1, 10) + 1
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

const samp_freqs = WeightVec{Float64,Vector{Float64}}[weights(rand(Dirichlet(10,1))) for i=1:10]
const energies = Float64[1,100,2,1,3,3,1,200,2,1]
function energy(obj::MySampler, block=1) # block is for AMWG
  return -log(energies[obj.curr])
  # -logpdf(Normal(obj.curr,1.0), obj.data) |> sum
end

# data = rand(5)
# data = [0.0,1.0,0.0,1.0]
data = Float64[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
thin = 10
burn = 5000
iters = int(5.1e5)
# burn = 200
# iters = 10000
loops = 10
correct = false
true_e = energies/sum(energies) .* [1:10;] |> sum
########################
# MH
########################
mh_rec = Float64[]
global mh
for i=1:loops
  mh = MHRecord(MySampler(1, 1, data), burn=burn, thin=thin)
  sample!(mh,iters)

  push!(mh_rec, posterior_e(identity, mh))
end
m,s = mean_and_std(mh_rec)
bias,stderr = mean_and_std(mh_rec - true_e)
@printf "MH    Mean: %6.3f, Std: %6.3f, Bias: %6.3f, Stderr: %6.3f \n" m s bias*1000 stderr*1000/sqrt(loops)
########################
# AMWG
########################
# numblocks = 1
# amwg_rec = Float64[]
# global amwg
# for i=1:loops
#   amwg = AMWGRecord(MySampler(1, 1, data), numblocks, burn=burn, thin=thin)
#   sample!(amwg,iters)
#
#   push!(amwg_rec, posterior_e(identity, amwg))
# end
# m,s = mean_and_std(amwg_rec)
# bias,stderr = mean_and_std(amwg_rec - true_e)
# @printf "AMWG  Mean: %6.3f, Std: %6.3f, Bias: %6.2f, Stderr: %6.2f \n" m s bias*1000 stderr*1000/sqrt(loops)
########################
# SAMC
########################
samc_rec = Float64[]
global samcsamp
for i=1:loops
  samcsamp = set_energy_limits(MySampler(1,1,data))
  # ^^ Convenience function to set energy limits
  samcsamp.stepscale = 10
  samcsamp.burn = burn
  samcsamp.thin = thin
  # samcsamp.grid = Float64[0,1,2,3,4,100,200]
  # samcsamp.grid = Float64[-6:0.5:1;]
  # samcsamp.grid = Float64[-5., -4.5, -1, -0.5, 0.0]
  # samcsamp.refden = Float64[length(samcsamp.grid)-1:-1:0;].^0
  # samcsamp.refden /= sum(samcsamp.refden)
  # samcsamp.counts = zeros(Int, length(samcsamp.grid))
  # samcsamp.thetas = zeros(Float64, length(samcsamp.grid))
  sample!(samcsamp, iters, correct=correct)
  push!(samc_rec, posterior_e(identity, samcsamp))
end
m,s = mean_and_std(samc_rec)
bias,stderr = mean_and_std(samc_rec - true_e)
@printf "SAMC  Mean: %6.3f, Std: %6.3f, Bias: %6.2f, Stderr: %6.2f \n" m s bias*1000 stderr*1000/sqrt(loops)
########################
# POP SAMC
########################
num_chains = 10
genfunc = _ -> MySampler(1, 1, data)

global psamcsamp
psamc_rec = Float64[]
for i=1:loops
  psamcsamp = set_energy_limits(genfunc, num_chains)
  # ^^ Convenience function to set energy limits
  psamcsamp.stepscale = 1
  psamcsamp.burn = burn
  psamcsamp.thin = thin
  sample!(psamcsamp, div(iters,num_chains), correct=correct)
  push!(psamc_rec, posterior_e(identity, psamcsamp))
end
m,s = mean_and_std(psamc_rec)
bias,stderr = mean_and_std(psamc_rec - true_e)
@printf "PSAMC Mean: %6.3f, Std: %6.3f, Bias: %6.2f, Stderr: %6.2f \n" m s bias*1000 stderr*1000/sqrt(loops)

# samcest = map(x->x[2],sort(collect(countmap(samcsamp.db))))/length(samcsamp.db)
