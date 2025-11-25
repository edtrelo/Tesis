using KernelDensity, Plots

function rolling_sum_lastn(v, n; p = 1)
    m = length(v)
    result = zeros(m)  # crea un vector del mismo tipo y tamaño
    for i in 1:m
        start_idx = max(1, i - (n-1))  # últimos 10 elementos incluyendo el actual
        result[i] = sum(v[start_idx:i].* 1/p)
    end
    return result
end

function rollmean(x::AbstractVector, k::Int; align=:center, na_rm=false)
    n = length(x)
    y = Vector{Union{Missing, Float64}}(undef, n)
    if k > n
        return fill(missing, n)
    end

    half = div(k,2)

    for i in 1:n
        if align == :center
            i1 = max(1, i - half)
            i2 = min(n, i + half)
        elseif align == :left
            i1 = i
            i2 = min(n, i + k - 1)
        elseif align == :right
            i1 = max(1, i - k + 1)
            i2 = i
        else
            error("align debe ser :center, :left o :right")
        end

        window = x[i1:i2]
        if length(window) == k
            y[i] = mean(window)
        else
            y[i] = na_rm ? missing : mean(window)
        end
    end
    return y
end

function plot_posterior_distribution(posterior_vector; interval = [], prior = nothing, n_points = 200)
    kde_post = kde(posterior_vector)
    interp_kde = InterpKDE(kde_post)
    # Función anónima
    posterior_pdf = x -> pdf(interp_kde, x)
    # intervalo de graficación 
    if length(interval) == 0
        interval  = range(minimum(β_post), maximum(β_post), length=n_points)
    end
    posterior_pdf_valores = posterior_pdf.(interval)
    p = plot(xβ, posterior_pdf, label="Posterior")
    if isnothing(prior)
    else
        p = plot!(interval, pdf.(prior, interval), lw=2, linestyle=:dash, label="Prior", color=:black)
    end
    return p
end