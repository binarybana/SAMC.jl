type SAMCRecord <: MCMC
    obj :: Sampler
    mapvalue
    mapenergy :: Float64
    db :: Vector{Any}
    db_theta :: Vector{Float64}

    counts :: Vector{Int}
    thetas :: Vector{Float64}
    energy_trace :: Vector{Float64}
    theta_trace :: Vector{Float64}

    grid :: AbstractArray{Float64}
    count_accept :: Int
    count_total :: Int
    iteration :: Int

    burn :: Int
    thin :: Int
    stepscale :: Float64
    delta :: Float64
    scale :: Float64

    nonempty :: Int

    refden :: Vector{Float64}
    refden_power :: Float64
end

SAMCRecord(obj::Sampler) = SAMCRecord(
    obj,record(obj),Inf,Array(typeof(record(obj)),0),Float64[],
    Int[],Float64[],Float64[],Float64[],
    0.0:0.0,0,0,1,
    1000,1,10000.0,1.0,1.0,
    0,
    Float64[],0.0)

function set_energy_limits(obj::Sampler; iters=1000, refden_power=0.0, verbose=0)
    record = SAMCRecord(obj)
    low = energy(obj)
    high = low
    oldenergy = low
    energyval = low
    while high > 1e90
        propose!(obj)
        low = high = oldenergy = energyval = energy(obj)
    end
    for i=1:iters
        propose!(obj)
        energyval = energy(obj)
        r = (oldenergy - energyval) / 3.0 # Higher temperature for exploration
        if r > 0.0 || rand() < exp(r) # Accept
            if energyval < low
                low = energyval
            elseif energyval > high
                high = energyval
            end
            oldenergy = energyval
            save!(obj)
        else
            reject!(obj)
        end
    end
    spread = high - low
    low = ifloor(low - (1.0*spread))
    high = iceil(high + (0.2*spread))
    spread = high - low
    record.scale = max(0.25, spread/100.0)
    if verbose > 0
      println("Done. Setting limits to ($low, $high)")
      println("Setting scale to $(record.scale)")
    end
    record.grid = low:record.scale:high
    record.refden = Float64[record.grid.len-1:-1:0;].^refden_power
    record.refden /= sum(record.refden)
    record.refden_power = refden_power
    record.counts = zeros(Int, length(record.grid))
    record.thetas = zeros(Float64, length(record.grid))
    record
end

function sample!(rec::SAMCRecord, iters::Int; temperature::Float64=1.0, beta::Float64=1.0, verbose=0, correct=true)
    oldenergy = energy(rec.obj)
    oldregion = clamp(searchsortedfirst(rec.grid, oldenergy), 1, length(rec.grid))
    if verbose>0
      println("Initial energy: $oldenergy")
    end

    for current_iter = rec.iteration:(rec.iteration+iters)
        rec.iteration += 1
        rec.delta = temperature * rec.stepscale / max(rec.stepscale, rec.iteration^beta)
        propose!(rec.obj)
        newenergy = energy(rec.obj)

        if newenergy < rec.mapenergy #I need to decide if I want this or not
            rec.mapenergy = newenergy
            rec.mapvalue = record(rec.obj)
        end

        ### Acceptance of new moves ###
        newregion = clamp(searchsortedfirst(rec.grid, newenergy), 1, length(rec.grid))
        r = rec.thetas[oldregion] - rec.thetas[newregion] + (oldenergy - newenergy)
        # r = rec.thetas[newregion] - rec.thetas[oldregion] + (oldenergy - newenergy)

        if r > 0.0 || rand() < exp(r) #Accept
            rec.counts[newregion] += 1
            rec.count_accept += 1
            oldregion = newregion
            oldenergy = newenergy
            save!(rec.obj)
        else #Reject
            rec.counts[oldregion] += 1
            reject!(rec.obj)
        end

        rec.thetas -= rec.delta*rec.refden#.*(rec.counts .> 0)
        rec.thetas[oldregion] += rec.delta

        push!(rec.energy_trace, oldenergy)
        push!(rec.theta_trace, rec.thetas[oldregion])

        if rec.iteration < rec.burn
            rec.nonempty = findfirst(rec.counts)
        elseif correct
            correction = rec.thetas[rec.nonempty]
            for i=1:length(rec.thetas)
                #if rec.counts[i] > 0
                    rec.thetas[i] -= correction
                #end
            end
        end
        rec.count_total += 1

        if rec.iteration >= rec.burn && rec.iteration%rec.thin == 0
            push!(rec.db, record(rec.obj))
            push!(rec.db_theta, rec.thetas[oldregion])
        end

        if rec.iteration % 10000 == 0 && verbose > 0
            @printf "Iteration: %8d, delta: %5.3f, best energy: %7f, current energy: %7f\n" rec.iteration rec.delta rec.mapenergy oldenergy
        end
    end
    #save_state_db(rec, temperature) #FIXME
    if verbose > 0
      println("Accepted samples: $(rec.count_accept)")
      println("Total samples: $(rec.count_total)")
      println("Acceptance: $(rec.count_accept/rec.count_total)")
    end
end

function posterior_e(f::Function, rec::SAMCRecord)
    N = length(rec.db)
    @assert N>0
    sumthetas = 0.0
    maxtheta = maximum(rec.db_theta)
    for i=1:N
        sumthetas += exp(rec.db_theta[i]-maxtheta)
    end

    sub = zero(f(rec.db[1]))
    for i=1:N
        sub += f(rec.db[i])*exp(rec.db_theta[i]-maxtheta)
    end
    sub /= sumthetas
    sub
end

function cum_posterior_e(f::Function, rec::SAMCRecord)
    N = length(rec.db)
    @assert N>0
    sumtheta = 0.0
    maxthetas = -Inf

    exres = zero(f(rec.db[1]))

    cum_post = Array(typeof(exres),0)
    fres = Array(typeof(exres),0)

    for i=1:N
        maxtheta = max(rec.db_theta[i],maxtheta)
        push!(fres, f(rec.db[i]))

        sumtheta = 0.0
        for j=1:i
            sumtheta += exp(rec.db_theta[j]-maxtheta)
        end
        sub = zero(eltype(fres))
        for j=1:i
            sub += fres[j]*exp(rec.db_theta[j]-maxtheta)
        end
        push!(cum_post, sub/sumtheta)
    end
    cum_post
end

function estimate_w(rec::SAMCRecord)
	C = logsumexp(rec.thetas[rec.counts.>0])
	v = sum(rec.refden[rec.counts.==0]) * 1/(length(rec.grid)-sum(rec.counts.==0))

	w = exp(rec.thetas - C) .* (rec.refden+v)
	w_filt = w[rec.counts.>0]
	psis = -log(w[rec.counts.>0])
  return w,w_filt,psis
end
