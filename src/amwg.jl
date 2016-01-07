type AMWGRecord <: MCMC
    obj :: Sampler
    blocks :: Int
    batchsize :: Int
    sigmas :: Vector{Float64}
    target :: Float64
    mapvalue
    mapenergy :: Float64
    db :: Vector{Any}

    block_accept :: Vector{Int}
    iterations :: Int

    burn :: Int
    thin :: Int
end

AMWGRecord(obj::Sampler, blocks; burn=0, thin=1) = AMWGRecord(obj, 
                                blocks, #Gibbs updates
                                50, # batchsize
                                ones(blocks), #sigmas
                                0.44, #target
                                record(obj), #mapvalue
                                Inf, #mapenergy
                                Array(typeof(record(obj)),0), #db
                                zeros(Int,blocks),1, #accept, iterations
                                burn,thin) #burn, thin


function sample!(rec::AMWGRecord, iters::Int; verbose=0, adjust="burnin")
    oldenergy = energy(rec.obj)
    if verbose>0
        println("Initial energy: $oldenergy")
    end

    for current_iter = rec.iterations:(rec.iterations+iters-1)
        sigma = randn(rec.blocks) .* rec.sigmas
        for i in 1:rec.blocks
            save!(rec.obj)
            #propose new object in gibbs block i, with sigma sigma[i]
            propose!(rec.obj, i, rec.sigmas[i])
            newenergy = energy(rec.obj, i) 
            if newenergy < rec.mapenergy
                rec.mapvalue = record(rec.obj)
                rec.mapenergy = newenergy
            end
            r = oldenergy - newenergy
            accept = rand() < exp(r)

            if (rec.iterations) % 100 == 0 && verbose > 3 && rec.iterations > rec.burn
                println("$(accept?"A":"R"): block: $i old: $oldenergy, new: $newenergy, diff: $(oldenergy-newenergy)")
            end
            ## Acceptance of new moves ###
            if accept
                rec.block_accept[i] += 1
                oldenergy = newenergy
            else
                reject!(rec.obj)
            end
        end

        # Adjust weights
        if current_iter % rec.batchsize == 0 && ((adjust == "burnin" && current_iter <= rec.burn) || adjust == "always")
            delta = min(0.3, (current_iter / rec.batchsize)^-0.9)
            verbose>2 && println("Sigmas before tuning: $(rec.sigmas)")
            verbose>2 && println("Delta: $delta")
            verbose>2 && println("Block_accept: $(rec.block_accept)")
            for i in 1:length(rec.sigmas)
                rec.sigmas[i] *= rec.block_accept[i] / rec.batchsize < rec.target ? exp(-delta) : exp(delta)
                rec.block_accept[i] = 0 # FIXME, is this right? I couldn't find it in mamba, but ow above doesn't make sense
            end
            verbose>2 && println("Sigmas after tuning: $(rec.sigmas)")
        end

        if rec.iterations >= rec.burn && rec.iterations%rec.thin == 0
            push!(rec.db, record(rec.obj))
        end
        
        if (rec.iterations) % 1000 == 0 && verbose>1
            @printf "Iteration: %8d, best energy: %7f, current energy: %7f\n" rec.iterations rec.mapenergy oldenergy
        end
        rec.iterations += 1
    end
    if verbose>0
        println("Accepted samples: $(rec.block_accept)")
        println("Acceptance ratios: $(rec.block_accept./(rec.iterations-rec.burn))")
    end
    return rec.block_accept./(rec.iterations-rec.burn)
end

##########################################################
# Functions for both AMWG, MH, and multiple chain MH
##########################################################

samples(recs::Vector{MHRecord}) = vcat([x.db for x in recs]...)
samples(rec::Union{MHRecord,AMWGRecord}) = rec.db

mapenergy(recs::Vector{MHRecord}) = minimum(Float64[x.mapenergy for x in recs])
mapenergy(rec::Union{MHRecord,AMWGRecord}) = rec.mapenergy

function mapvalue(recs::Vector{MHRecord})
  i = indmin(Float64[x.mapenergy for x in recs])
  recs[i].mapvalue
end

mapvalue(rec::Union{MHRecord,AMWGRecord}) = rec.mapvalue

function posterior_e(f::Function, recs::Union{Vector{MHRecord},Vector{AMWGRecord}})
  subN = length(recs[1].db)
  N = sum(map(x->length(x.db), recs))
  K = length(recs)
  @assert N>0
  @assert K>0
  @assert N == subN*K
  sub = zero(f(recs[1].db[1]))
  for chain=1:K,i=1:subN
      sub += f(recs[chain].db[i])
    end
  sub /= N
end

function posterior_e(f::Function, rec::Union{MHRecord,AMWGRecord})
    N = length(rec.db)
    @assert N>0
    sub = zero(f(rec.db[1]))
    for i=1:N
        sub += f(rec.db[i])
      end
    sub /= N
end

function cum_posterior_e(f::Function, rec::Union{MHRecord,AMWGRecord})
    N = length(rec.db)
    @assert N>0
    sumthetas = 0.0
    maxthetas = -Inf

    sub = zero(f(rec.db[1]))
    cum_post = Array(typeof(sub),0)
    for i=1:N
        sub += f(rec.db[i])
        push!(cum_post, sub/i)
    end
    cum_post
end
