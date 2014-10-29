type SAMCRecord <: MCMC
    obj :: Sampler
    mapvalue :: Sampler
    mapenergy :: Float64
    db :: Any
    counts :: Vector{Int}
    thetas :: Vector{Float64}
    grid :: Ranges{Float64}

    count_accept :: Int
    count_total :: Int
    iteration :: Int

    burn :: Int
    thin :: Int
    stepscale :: Float64
    delta :: Float64
    scale :: Float64

    refden :: Vector{Float64}
    refden_power :: Float64
end

SAMCRecord(obj::Sampler) = SAMCRecord(obj,obj,0.0,nothing,Int[],Float64[],0.0:0.0,0,0,1,0,1,10000.0,1.0,1.0,Float64[],0.0)

function set_energy_limits(obj::Sampler, iters=1000, refden_power=0.0)
    record = SAMCRecord(obj)
    low = energy(obj) 
    high = low
    oldenergy = low
    energyval = low
    while high > 1e90
        propose(obj)
        low = high = oldenergy = energyval = energy(obj)
    end
    oldobj = copy(obj)
    for i=1:iters
        copy!(oldobj, obj)
        propose(obj)
        energyval = energy(obj)
        r = (oldenergy - energyval) / 3.0 # Higher temperature for exploration
        if r > 0.0 || rand() < exp(r) # Accept
            if energyval < low
                low = energyval
            elseif energyval > high
                high = energyval
            end
            oldenergy = energyval
        else
            copy!(obj, oldobj) # Reject
        end
    end
    spread = high - low
    low = ifloor(low - (0.6*spread))
    high = iceil(high + (0.2*spread))
    println("Done. Setting limits to ($low, $high)")
    spread = high - low
    record.scale = max(0.25, spread/100.0)
    println("Setting scale to $(record.scale)")
    record.grid = low:record.scale:high
    record.refden = Float64[record.grid.len-1:-1:0].^refden_power
    record.refden /= sum(record.refden)
    record.refden_power = refden_power
    record.counts = zeros(Int, record.grid.len)
    record.thetas = zeros(Float64, record.grid.len)
    record
end

function clear(rec::SAMCRecord)
    rec.db = nothing
    rec.delta = 1.0
    rec.iteration = 0
    rec.count_accept = 0
    rec.count_total = 0
end

function sample(rec::SAMCRecord, iters::Int, temperature::Float64=1.0; verbose=0)
    oldenergy = energy(rec.obj)
    oldregion = clamp(searchsortedfirst(rec.grid, oldenergy), rec.grid.start, rec.grid.len)
    dbsize = div(rec.iteration + iters - rec.burn, rec.thin)
    dbsize = dbsize > 0 ? dbsize : 0
    println("Initial energy: $oldenergy")

    for current_iter = rec.iteration:(rec.iteration+iters)
        rec.iteration += 1
        rec.delta = temperature * rec.stepscale / max(rec.stepscale, rec.iteration)
        propose(rec.obj)
        newenergy = energy(rec.obj)

        if newenergy < rec.mapenergy #I need to decide if I want this or not
            rec.mapenergy = newenergy
            rec.mapvalue = deepcopy(rec.obj)
        end

        ### Acceptance of new moves ###
        newregion = clamp(searchsortedfirst(rec.grid, newenergy), rec.grid.start, rec.grid.len)
        r = rec.thetas[oldregion] - rec.thetas[newregion] + (oldenergy - newenergy)
        if r > 0.0 || rand() < exp(r) #Accept
            rec.counts[newregion] += 1
            rec.count_accept += 1
            oldregion = newregion
            oldenergy = newenergy
        else #Reject
            rec.counts[oldregion] += 1
            reject(rec.obj)
        end
        rec.count_total += 1
        rec.thetas -= rec.delta*rec.refden
        rec.thetas[oldregion] += rec.delta

        #if rec.iteration >= rec.burn && rec.iteration%rec.thin == 0
            #save_iter_db(rec.thetas[oldregion], oldenergy,
            #div(rec.iteration-rec.burn, rec.thin)) # FIXME
        #end
        
        if rec.iteration % 10000 == 0
            @printf "Iteration: %8d, delta: %5.2f, best energy: %7f, current energy: %7f\n" rec.iteration rec.delta rec.mapenergy oldenergy
        end
    end
    #save_state_db(rec, temperature) #FIXME
    println("Accepted samples: $(rec.count_accept)")
    println("Total samples: $(rec.count_total)")
    println("Acceptance: $(rec.count_accept/rec.count_total)")
end