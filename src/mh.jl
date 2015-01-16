type MHRecord <: MCMC
    obj :: Sampler
    mapvalue
    mapenergy :: Float64
    db :: Vector{Any}

    count_accept :: Int
    count_total :: Int
    iteration :: Int

    burn :: Int
    thin :: Int

    scheme_propose :: Dict{Int,Int}
    scheme_accept :: Dict{Int,Int}
end

MHRecord(obj::Sampler; burn=0, thin=1) = MHRecord(obj,
                                record(obj), #mapvalue
                                Inf, #mapenergy
                                Array(typeof(record(obj)),0), #db
                                0,0,1, #accept, total, iteration
                                burn,thin, #burn, thin
                                (Int=>Int)[],
                                (Int=>Int)[])

#import Distributions.sample

function sample!(recs::Vector{MHRecord}, iters::Int; verbose=0)
  for rec in recs
    sample!(rec, iters, verbose=verbose)
  end
end

function sample!(rec::MHRecord, iters::Int; verbose=0)
    oldenergy = energy(rec.obj)
    if verbose > 0
        println("Initial energy: $oldenergy")
    end

    for current_iter = rec.iteration:(rec.iteration+iters-1)
        save!(rec.obj)
        scheme = propose!(rec.obj)
        rec.scheme_propose[scheme] = get(rec.scheme_propose, scheme, 0) + 1
        newenergy = energy(rec.obj)

        if newenergy < rec.mapenergy #I need to decide if I want this or not
            rec.mapenergy = newenergy
            rec.mapvalue = record(rec.obj)
        end

        ### Acceptance of new moves ###
        r = oldenergy - newenergy
        if r > 0.0 || rand() < exp(r) #Accept
            rec.count_accept += 1
            oldenergy = newenergy
            rec.scheme_accept[scheme] = get(rec.scheme_accept, scheme, 0) + 1
            if verbose > 1
                println("A: old: $oldenergy, new: $newenergy, diff: $(oldenergy-newenergy)")
            end
        else #Reject
            reject!(rec.obj)
            if verbose > 1
                println("R: old: $oldenergy, new: $newenergy, diff: $(oldenergy-newenergy)")
            end
        end
        rec.count_total += 1

        if rec.iteration >= rec.burn && rec.iteration%rec.thin == 0
            push!(rec.db, record(rec.obj))
        end

        if (rec.iteration) % 1000 == 0 && verbose > 0
            @printf "Iteration: %8d, best energy: %7f, current energy: %7f\n" rec.iteration rec.mapenergy oldenergy
        end
        rec.iteration += 1
    end
    if verbose > 0
        println("Accepted samples: $(rec.count_accept)")
        println("Total samples: $(rec.count_total)")
        println("Acceptance: $(rec.count_accept/rec.count_total)")
    end
end

