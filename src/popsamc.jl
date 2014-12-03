type PopSAMCRecord <: MCMC
    objs :: Vector{Sampler}
    mapvalue :: Sampler
    mapenergy :: Float64
    dbs :: Vector{Vector{Any}}

    dbs_theta :: Vector{Vector{Float64}}

    counts :: Vector{Int}
    thetas :: Vector{Float64}
    energy_traces :: Vector{Vector{Float64}}
    theta_traces :: Vector{Vector{Float64}}

    grid :: Range{Float64}
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

PopSAMCRecord(genfunc::Function, k::Int) = PopSAMCRecord(map(genfunc,1:k),genfunc(1),Inf,map(x->Any[],1:k),
                                        map(x->Float64[],1:k),
                                        Int[],Float64[],map(x->Float64[],1:k),map(x->Float64[],1:k),
                                        0.0:0.0,0,0,1,
                                        1000,1,10000.0,1.0,1.0,
                                        0,
                                        Float64[],0.0)

function set_energy_limits(genfunc::Function, k::Int; iters=1000, refden_power=0.0)
    record = PopSAMCRecord(genfunc, k)
    oldenergies = map(energy,record.objs) 
    low = minimum(oldenergies)
    high = maximum(oldenergies)
    for i=1:iters
        for j=1:k
            propose!(record.objs[j])
            energyval = energy(record.objs[j])
            r = (oldenergies[j] - energyval) / 3.0 # Higher temperature for exploration
            if r > 0.0 || rand() < exp(r) # Accept
                if energyval < low
                    low = energyval
                elseif energyval > high
                    high = energyval
                end
                oldenergies[j] = energyval
                save!(record.objs[j])
            else
                reject!(record.objs[j])
            end
        end
    end
    spread = high - low
    low = ifloor(low - (1.0*spread))
    high = iceil(high + (0.2*spread))
    println("Done. Setting limits to ($low, $high)")
    spread = high - low
    record.scale = max(0.25, spread/100.0)
    println("Setting scale to $(record.scale)")
    record.grid = low:record.scale:high
    record.refden = Float64[record.grid.len-1:-1:0].^refden_power
    record.refden /= sum(record.refden)
    record.refden_power = refden_power
    record.counts = zeros(Int, length(record.grid))
    record.thetas = zeros(Float64, length(record.grid))
    record
end

function sample(rec::PopSAMCRecord, iters::Int; temperature::Float64=1.0, beta::Float64=1.0, verbose=0)
    if rec.grid == 0.0:0.0
        throw(Exception("You must set_energy_limits on your PopSAMCRecord before calling sample"))
    end

    oldenergies = map(energy,rec.objs) 
    oldregions = map(x->clamp(searchsortedfirst(rec.grid, x), 1, length(rec.grid)), oldenergies)
    k = length(rec.objs)

    print("Initial energies: ")
    for i=1:k
        @printf "%5.2f," oldenergies[i]
    end
    println("")

    for current_iter = rec.iteration:(rec.iteration+iters)
        rec.iteration += 1
        rec.delta = temperature * rec.stepscale / max(rec.stepscale, rec.iteration^beta)
        for chain=1:k
            propose!(rec.objs[chain])
            newenergy = energy(rec.objs[chain])

            if newenergy < rec.mapenergy #I need to decide if I want this or not
                rec.mapenergy = newenergy
                rec.mapvalue = deepcopy(rec.objs[chain])
            end

            ### Acceptance of new moves ###
            newregion = clamp(searchsortedfirst(rec.grid, newenergy), 1, length(rec.grid))
            r = rec.thetas[oldregions[chain]] - rec.thetas[newregion] + (oldenergies[chain] - newenergy)

            if r > 0.0 || rand() < exp(r) #Accept
                rec.counts[newregion] += 1
                rec.count_accept += 1
                save!(rec.objs[chain])
                oldenergies[chain] = newenergy
                oldregions[chain] = newregion
            else #Reject
                rec.counts[oldregions[chain]] += 1
                reject!(rec.objs[chain])
            end
        end

        rec.thetas -= rec.delta*rec.refden #.*(rec.counts .> 0)
        for region in oldregions
            rec.thetas[region] += 2*rec.delta*rec.refden[region]/k
        end

        if rec.iteration < rec.burn
            rec.nonempty = findfirst(rec.counts)
        else
            correction = rec.thetas[rec.nonempty]
            for i=1:length(rec.thetas)
                #if rec.counts[i] > 0
                    rec.thetas[i] -= correction
                #end
            end
        end

        for chain=1:k
            push!(rec.energy_traces[chain], oldenergies[chain])
            push!(rec.theta_traces[chain], rec.thetas[oldregions[chain]])
        end

        if rec.iteration >= rec.burn && rec.iteration%rec.thin == 0
            for chain=1:k
                push!(rec.dbs[chain], deepcopy(rec.objs[chain]))
                push!(rec.dbs_theta[chain], rec.thetas[oldregions[chain]])
            end
        end
        rec.count_total += k
        
        if rec.iteration % 10000 == 0
            @printf "Iteration: %8d, delta: %5.3f, best energy: %7f, current first energy: %7f\n" rec.iteration rec.delta rec.mapenergy oldenergies[1]
        end
    end
    println("Accepted samples: $(rec.count_accept)")
    println("Total samples: $(rec.count_total)")
    println("Acceptance: $(rec.count_accept/rec.count_total)")
end

function posterior_e(f::Function, rec::PopSAMCRecord)
    K = length(rec.dbs) 
    N = length(rec.dbs[1])
    @assert K>0
    @assert N>0
    sumthetas = 0.0
    maxthetas = -Inf
    for k=1:K
        maxthetas = max(maximum(rec.dbs_theta[k]), maxthetas)
    end
    for k=1:K,i=1:N
        sumthetas += exp(maxthetas-rec.dbs_theta[k][i])
    end
    
    sub = zero(f(rec.dbs[1][1]))
    for k=1:K,i=1:N
        sub += f(rec.dbs[k][i])*exp(maxthetas-rec.dbs_theta[k][i])
    end
    sub /= sumthetas
    sub
end

function cum_posterior_e(f::Function, rec::PopSAMCRecord)
    K = length(rec.dbs) 
    N = length(rec.dbs[1])
    @assert K>0
    @assert N>0
    sumthetas = 0.0
    maxthetas = -Inf

    exres = zero(f(rec.dbs[1][1]))

    cum_post = Array(typeof(exres),0)
    fres = [Array(typeof(exres),0) for _=1:K]

    for i=1:N
        for k=1:K
            maxthetas = max(rec.dbs_theta[k][i], maxthetas)
            push!(fres[k], f(rec.dbs[k][i]))
        end
        sumthetas = 0.0
        for k=1:K,j=1:i
            sumthetas += exp(maxthetas-rec.dbs_theta[k][j])
        end
        sub = zero(eltype(fres[1]))
        for k=1:K,j=1:i
            sub += fres[k][j]*exp(maxthetas-rec.dbs_theta[k][j])
        end
        push!(cum_post, sub/sumthetas)
    end
    cum_post
end

