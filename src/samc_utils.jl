using PyPlot

function plotsamc(s::PopSAMCRecord, burn=1)
    energy = s.grid
    theta = s.thetas
    counts = s.counts
    energy_traces = s.energy_traces
    theta_traces = s.theta_traces
    dbs_theta = s.dbs_theta

    if s.burn > 0
        println("Using burn in SAMC")
        burn = s.burn
    end

    rows = 4
    cols = 2

    figure()
    subplot(rows, cols, 1)
    plot(energy, theta, "k.")
    title("Region's theta values")
    ylabel("Theta")
    xlabel("Energy")

    subplot(rows, cols, 2)
    plot(energy, counts, "k.")
    title("Region's Sample Counts")
    ylabel("Count")
    xlabel("Energy")
    yl,yh = ylim()
    ylim(0,yh)

    subplot(rows, cols, 3)
    @show length(burn:(length(energy_traces[1])+burn))
    @show length(energy_traces[1])
    #plot(burn:(length(energy_trace)+burn), energy_trace, "k.")
    k=length(s.objs)
    for i=1:k
        plot(energy_traces[i], "k.")
    end
    title("Energy Trace")
    ylabel("Energy")
    xlabel("Iteration")
    axvline(burn)

    subplot(rows, cols, 4)
    #plot(burn:(length(theta_trace)+burn), theta_trace, "k.")
    for i=1:k
        plot(theta_traces[i], "k.")
    end
    ylabel("Theta Trace")
    xlabel("Iteration")
    axvline(burn)
        
    subplot(rows, cols, 5)
    theta_trace = vcat(theta_traces...)
    part = exp(theta_trace[burn:end] - maximum(theta_trace[burn:end]))
    plt.hist(part, log=true, bins=100)
    xlabel("exp(theta - theta_max)")
    ylabel("Number of samples at this value")
    title("Histogram of normalized sample thetas from $(length(theta_trace)) iterations")

    subplot(rows, cols, 6)
    plt.hist(part, weights=part, bins=50)
    xlabel("exp(theta - theta_max)")
    ylabel("Amount of weight at this value")

    subplot(rows, cols, 7)
    dbs_theta = vcat(dbs_theta...)
    part = exp(dbs_theta - maximum(dbs_theta))
    plt.hist(part, log=true, bins=100)
    xlabel("exp(theta - theta_max)")
    ylabel("Number of samples at this value")
    title("Histogram of normalized sample thetas from $(length(dbs_theta)) iterations")

    subplot(rows, cols, 8)
    plt.hist(part, weights=part, bins=50)
    xlabel("exp(theta - theta_max)")
    ylabel("Amount of weight at this value")
end

function plotsamc(s::SAMCRecord, burn=1)
    energy = s.grid
    theta = s.thetas
    counts = s.counts
    energy_trace = s.energy_trace
    theta_trace = s.theta_trace
    if s.burn > 0
        println("Using burn in SAMC")
        burn = s.burn
    end


    rows = 3
    cols = 2

    figure()
    subplot(rows, cols, 1)
    plot(energy, theta, "k.")
    title("Region's theta values")
    ylabel("Theta")
    xlabel("Energy")

    subplot(rows, cols, 2)
    plot(energy, counts, "k.")
    title("Region's Sample Counts")
    ylabel("Count")
    xlabel("Energy")
    yl,yh = ylim()
    ylim(0,yh)

    subplot(rows, cols, 3)
    @show length(burn:(length(energy_trace)+burn))
    @show length(energy_trace)
    #plot(burn:(length(energy_trace)+burn), energy_trace, "k.")
    plot(energy_trace, "k.")
    title("Energy Trace")
    ylabel("Energy")
    xlabel("Iteration")
    axvline(burn)

    subplot(rows, cols, 4)
    #plot(burn:(length(theta_trace)+burn), theta_trace, "k.")
    plot(theta_trace, "k.")
    ylabel("Theta Trace")
    xlabel("Iteration")
    axvline(burn)
        
    subplot(rows, cols, 5)
    part = exp(theta_trace[burn:end] - maximum(theta_trace[burn:end]))
    plt.hist(part, log=true, bins=100)
    xlabel("exp(theta - theta_max)")
    ylabel("Number of samples at this value")
    title("Histogram of normalized sample thetas from $(length(theta_trace)) iterations")

    subplot(rows, cols, 6)
    plt.hist(part, weights=part, bins=50)
    xlabel("exp(theta - theta_max)")
    ylabel("Amount of weight at this value")
end

