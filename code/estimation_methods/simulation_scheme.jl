# Estimación por esquema de simulación
using Distributions
using Plots
using DifferentialEquations
using Distributions
using QuadGK
using Interpolations
using Statistics
using Distributions
using BlackBoxOptim
using CSV
using DataFrames
using Optim
using Printf
using Colors

script_dir = @__DIR__  
include(joinpath(script_dir,  "../utils/SIS_model.jl"))

data = CSV.read(
    joinpath(script_dir, "../../data/datos_limpios/modelo_SIS/espanadata.csv"), DataFrame
)

I = data.prevalence
N = mean(data.population)
γ = 1/(21/(365.25*10))
μ = 1/(8.267-1.5)
I = I / N

function logsumexp(x)
    isempty(x) && return -Inf
    max_x = maximum(x)
    return max_x + log(sum(exp.(x .- max_x)))
end

# Simulation-Scheme Estimation
function neg_log_lik_SSE(obs; Δt=0.1)
    
    T, d = size(obs)

    function neg_log_lik(θ::Vector)
        β, σ = θ
        p = [β, γ, μ, σ, N, 1.0] 

        l = 0.0

        @inbounds for i in 2:T
            X_prev = obs[i-1, :]
            Y_curr = obs[i, :]

            try 
                Σ = g(X_prev, p, 0.0) * g(X_prev, p, 0.0)' * Δt
                μ_drift = X_prev + f(X_prev, p, 0.0) * Δt
                D = MvNormal(μ_drift, Σ)
                l += logpdf(D, Y_curr)

            catch e 
                return 1e12 
            end
        end
        
        return -l
    end

    return neg_log_lik
end

function neg_log_lik_SMLE(obs, Δt, M, S, noise_paths)
    T = size(obs, 1)
    h = Δt / M # Tamaño de paso interno

    if size(noise_paths) != (M - 1, S)
         error("Dimensiones de 'noise_paths' incorrectas. Se espera ($(M-1), $(S)).")
    end

    scaled_noise = noise_paths .* sqrt(h)

    function neg_log_lik(θ::Vector)
    
        # Parámetros fijos
        p = [θ[1], γ, μ, θ[2], N, 1.0] 
        
        ℓ = 0.0

        @inbounds @simd for i in 2:T
            y_ti = obs[i-1, :]
            y_ti_plus_1 = obs[i, :] 

            log_q_s_list = zeros(S) # Almacena log(phi)

            Threads.@threads for s in 1:S
                X_curr = copy(y_ti) # X^{(s)}_{0} <- y_{t_i}
                
                # 1. M-1 pasos de Euler-Maruyama (Simulación)
                for k in 1:M-1
                    dW = scaled_noise[k, s] 
                    X_curr += f(X_curr, p, 0.0) .* h .+ g(X_curr, p, 0.0) .* dW
                end

                # X_curr es ahora z_s (el penúltimo valor simulado)
                z_s = X_curr
                
                # 2. Paso final (Densidad Normal)
                # La media para el último paso (aproximación SLE)
                μ_drift = f(z_s, p, 0.0) .* h
                Σ_drift = g(z_s, p, 0.0)*g(z_s, p, 0.0)' * h
                
                # La media completa: z_s + f(z_s)h
                μ_pred = z_s .+ μ_drift
                
                try
                    D = MvNormal(μ_pred, Σ_drift) 
                    log_q_s_list[s] = logpdf(D, y_ti_plus_1)
                catch
                    # Si logpdf falla (ej. Σ es muy pequeño), penalizamos con un valor seguro.
                    log_q_s_list[s] = -1e12 
                end
            end
            
            # 3. Log-Verosimilitud Simulada Acumulada (LogSumExp trick)
            # log(q_M_S) = log( (1/S) * sum(exp(log(phi))) ) = LogSumExp(log(phi)) - log(S)
            log_q_M_S = logsumexp(log_q_s_list) - log(S)

            ℓ += log_q_M_S
        end
        
        return -ℓ
    end

    return neg_log_lik
end

function get_jacobian_penalty(z::AbstractVector{T}, lower_bounds::Vector{Float64}, upper_bounds::Vector{Float64}) where T <: Real
    
    # Verificar que el número de límites coincida con la dimensión de los parámetros
    if length(lower_bounds) != length(upper_bounds)
        error("Los vectores de límites inferior y superior deben tener la misma longitud.")
    end

    # Sigmoide: 1 / (1 + exp(-z))
    sigmoid(z) = 1.0 / (1.0 + exp(-z))
    
    # Calcular la amplitud (b - a) fuera del bucle de optimización
    amplitude = upper_bounds .- lower_bounds

    # 1. Transformación Logit Inversa (mapea z a (a_i, b_i))
    # θ = a + (b - a) * sigmoid(z)
    s = sigmoid.(z)
    θ = lower_bounds .+ amplitude .* s
    
    # 2. Corrección del Jacobiano 
        
    # log(J_ii) = log(b_i - a_i) + log(sigmoid(z_i)) + log(1 - sigmoid(z_i))
        
    # log(b_i - a_i) se calcula a partir de 'amplitude'
    log_jacobian_term = log.(amplitude) .+ log.(s) .+ log.(1.0 .- s)
        
    # La corrección es la suma de los logaritmos negativos de los términos diagonales
    jacobian_penalty = sum(log_jacobian_term)

    return jacobian_penalty

end
          
sigmoid(z) = 1.0 / (1.0 + exp(-z))

calculate_aic(nll::Float64, k::Int) = 2.0 * k + 2.0 * nll

function logit(θ::Float64, a::Float64, b::Float64)::Float64
    p = (θ - a) / (b - θ)
    return log(p)
end

#θ0 = [8.753e-9, 1.654537e-8]  # β, σ
θ0 = [4.463e-07, 1.653283e-8]
θ0 = [4.403e-06, 1.653261e-08]
upper_limits = [1.0, 1.0]
lower_limits = [0.0, 0.0] 
K = length(θ0)
optim_options = Optim.Options(g_tol = 1e-6, show_trace = false);
z0 = [logit(θ0[k], lower_limits[k], upper_limits[k]) for k in 1:K]

# estimación por SSE
neg_log_lik_SSE_func = neg_log_lik_SSE(reshape(I, (length(I), 1)), Δt=0.1)
sse_cost_func = z::Vector -> neg_log_lik_SSE_func(lower_limits .+ (upper_limits .- lower_limits) .* sigmoid.(z)) - get_jacobian_penalty(
    z, lower_limits, upper_limits
)

result_sse = optimize(sse_cost_func, z0, NelderMead(), optim_options, autodiff = :forward)
z_sse = Optim.minimizer(result_sse)
nll_sse = Optim.minimum(result_sse) + get_jacobian_penalty(z_sse, lower_limits, upper_limits)
θ_sse = lower_limits .+ (upper_limits .- lower_limits) .* (1.0 ./ (1.0 .+ exp.(-z_sse)))

aic_sse = calculate_aic(nll_sse, K)

sim = simulate_sis_sde(θ_sse[1], γ, μ, θ_sse[2], [I[1]*N], N, 1.0, saveat = 0.1, dt = 0.001, normalized = 0.0)
plot(sim)
plot!(collect(0:0.1:1.0), I.*N)

# estimación por SMLE
M = [10, 100, 250, 1000, 5000, 10000]
S = [50, 250, 1000, 2500]
J = 50  # Número de ejecuciones para análisis estadístico

M_max = maximum(M)
S_max = maximum(S)

const NOISE_PATHS_GLOBAL = randn(M_max - 1, S_max) 

noise_subset = view(NOISE_PATHS_GLOBAL, 1:(M_max - 1), 1:S_max) 

total_m_s_combinations = length(M) * length(S)
total_smle_tasks = J *total_m_s_combinations
all_smle_results = Vector{Dict{String, Any}}(undef, total_smle_tasks)

@time Threads.@threads for l in 1:total_smle_tasks

    j = ceil(Int, l / total_m_s_combinations) 
    idx = l - (j - 1) * total_m_s_combinations
    m_idx = ceil(Int, idx / length(S))
    s_idx = idx - (m_idx - 1) * length(S)

    m = M[m_idx]
    s = S[s_idx]

    noise_subset = view(NOISE_PATHS_GLOBAL, 1:(m - 1), 1:s) 

    neg_log_lik_SMLE_func = neg_log_lik_SMLE(reshape(I .+ 0.0, (length(I), 1)), 0.1, m, s, noise_subset)
    smle_cost_func = z::Vector -> neg_log_lik_SMLE_func(lower_limits .+ (upper_limits .- lower_limits) .* sigmoid.(z)) - 
        get_jacobian_penalty(z, lower_limits, upper_limits)

    result_smle = optimize(smle_cost_func, z0, LBFGS(), optim_options, autodiff = :forward)
    z_smle = Optim.minimizer(result_smle)
    nll_smle = Optim.minimum(result_smle) + get_jacobian_penalty(z_smle, lower_limits, upper_limits)
    θ_smle = lower_limits .+ (upper_limits .- lower_limits) .* (1.0 ./ (1.0 .+ exp.(-z_smle)))

    if nll_smle == Inf
        println("Falló el LBFGS")
        break
    end

    all_smle_results[l] = Dict(
        "j" => j,
        "M" => m,
        "S" => s,
        "β" => θ_smle[1],
        "σ" => θ_smle[2],
        "NLL" => nll_smle,
        "AIC" => calculate_aic(nll_smle, K)
    )

end 
# tardó 44 minutos

# TARDÓ 8752 SEGUNDOS


all_smle_results
smle_df = DataFrame(all_smle_results)

CSV.write(joinpath(script_dir, "../../data/resultados/modelo_SIS/all_smle_results.csv"), smle_df)

smle_df = CSV.read(joinpath(script_dir, "../../data/resultados/modelo_SIS/all_smle_results.csv"), DataFrame)

smle_summary = combine(
    groupby(smle_df, [:S, :M]),
    # Medias de los estimadores y métricas
    :β => mean => :beta,
    :σ => mean => :sigma,
    :NLL => mean => :NLL,
    :AIC => mean => :AIC,
    # Desviación estándar (Std)
    :β => std => :Std_beta,
    :σ => std => :Std_sigma,
    # NUEVO: Cálculo de la longitud del Intervalo de Confianza (95%)
    :β => (x -> quantile(x, 0.975) - quantile(x, 0.025)) => :CI_Length_beta,
    :σ => (x -> quantile(x, 0.975) - quantile(x, 0.025)) => :CI_Length_sigma,
    # Conteo (debe ser igual a J)
    nrow => :Count
)

filter(smle_df, smle_df[""])

smle_summary[!, :method] = [@sprintf("SMLE(%d, %d)", row.S, row.M) for row in eachrow(smle_summary)]

# Reordenar las columnas para el formato de salida deseado (con las medias primero)
smle_summary = smle_summary[!, [:method, :beta, :sigma, :AIC, :NLL]]


z_sse = Optim.minimizer(result_sse)
nll_sse = Optim.minimum(result_sse) + get_jacobian_penalty(z_sse, lower_limits, upper_limits)
θ_sse = lower_limits .+ (upper_limits .- lower_limits) .* (1.0 ./ (1.0 .+ exp.(-z_sse)))

aic_sse = calculate_aic(nll_sse, K)


sse_summary = DataFrame(
    method = "SSE",
    beta = θ_sse[1],
    sigma = θ_sse[2],
    NLL = nll_sse,
    AIC = aic_sse,
)

final_summary = vcat(smle_summary, sse_summary, cols=:union)
sort!(final_summary, :AIC)

final_summary



time_points = collect(0.0:0.1:1.0)
nSims = 100

# --- Funciones auxiliares ---
is_smle(method) = occursin("SMLE", method)

function extract_MS(method_str)
    nums = parse.(Int, split(split(method_str, '(')[2][1:end-1], ","))
    return nums[1], nums[2]
end

# Listas de valores
M_values = unique([extract_MS(row.method)[2] for row in eachrow(final_summary) if is_smle(row.method)])
S_values = unique([extract_MS(row.method)[1] for row in eachrow(final_summary) if is_smle(row.method)])
minS, maxS = minimum(S_values), maximum(S_values)

# Asignar markers por M
marker_list = [:square, :diamond, :utriangle, :cross]
marker_map = Dict(M => marker_list[i] for (i, M) in enumerate(sort(M_values)))

# Gradiente azul para S
function color_from_S(S)
    α = (S - minS) / (maxS - minS + 1e-9)
    return RGB((1-α)*0.7, (1-α)*0.85, (1-α)*1.0 + α*0.4) # azul claro→oscuro
end

function plot_comparison_with_zoom(final_summary::DataFrame)
    
    # 1. GENERAR EL GRÁFICO PRINCIPAL (t=0 a t=10)
    
    p_main = plot(time_points, I .* N, label="Datos observados", 
                  lw=2, marker=:circle, color=:orange,
                  title="Ajuste de Estimadores y Convergencia de Modelos",
                  xlabel="Tiempo",
                  ylabel="Prevalencia Absoluta (I)",
                  legend=:topleft,
                  size=(800, 600),
                  ylim=(0, 1000))
    
    M_plotted_main = Set{Int}() 

    # --- LOOP PRINCIPAL (Genera todas las trayectorias para el main plot) ---
    for row in eachrow(final_summary)

        if !is_smle(row.method) && row.method != "SSE"
            continue
        end

        # REEMPLAZO DE LA LÓGICA DE SIMULACIÓN DIRECTA (nSims=100)
        nSims = 100
        time_points = collect(0.0:0.1:1.0) # t=0 a t=10, 11 puntos
        I_paths = zeros(length(time_points), nSims) # (Tiempo x Simulación)
        
        for t_idx in 1:nSims
            # NOTA: Se asume que simulate_sis_sde retorna una estructura sol
            sol = simulate_sis_sde(row.beta, γ, μ, row.sigma,
                                   [I[1]*N], N, 1.0,
                                   saveat=0.1, dt=0.001, normalized=0.0)
            # Almacenamos el vector de la solución en una columna
            I_paths[:, t_idx] = first.(sol.u) 
        end

        # Cálculo de la media sobre la dimensión de las simulaciones (Dim 2)
        mean_path = mean(I_paths, dims=2)[:] 

        # Configuración de estilo
        if row.method == "SSE"
            plot!(p_main, time_points, mean_path, lw=2, color=:red, label="SSE", marker=:circle)
            continue
        end

        # --- SMLE ---
        S_val, M_val = extract_MS(row.method)
        color = color_from_S(S_val)
        marker = marker_map[M_val]

        # Etiquetado (Solo la primera M)
        if M_val in M_plotted_main
            label = "" 
        else
            label = "SMLE(M=$M_val)"
            push!(M_plotted_main, M_val)
        end
        
        # Ploteo de la línea principal
        plot!(p_main, time_points, mean_path,
              lw=2, color=color,
              marker=marker, markersize = 3, label=label)
    end
    # --- FIN DEL LOOP PRINCIPAL ---

    # 2. CREAR EL GRÁFICO DE INSERCIÓN (ZOOM: t=5 a t=9)
    
    zoom_indices = 6:11
    zoom_time = time_points[zoom_indices]
    
    p_zoom = plot(size=(400, 400), margin=10 * Plots.mm) 
    
    # Ploteo de los datos observados en el zoom
    plot!(p_zoom, zoom_time, (I .* N)[zoom_indices], label="",
          lw=2, marker=:circle, color=:orange, xlims=(.58, 1.02), 
          ylims=(minimum((I .* N)[zoom_indices]) * 0.9, maximum((I .* N)[zoom_indices]) * 1.1))

    # Re-ejecutar el loop para trazar las líneas SMLE/SSE en el plot de zoom
    # Aquí es necesario repetir la simulación para tener el mean_path_full en el scope
    M_plotted_zoom = Set{Int}()
    for row in eachrow(final_summary)

        if !is_smle(row.method) && row.method != "SSE"
            continue
        end
        
        # --- REPETICIÓN DE LA SIMULACIÓN PARA EL ZOOM ---
        nSims = 100
        I_paths = zeros(length(time_points), nSims)
        for t_idx in 1:nSims
            sol = simulate_sis_sde(row.beta, γ, μ, row.sigma,
                                [I[1]*N], N, 1.0,
                                saveat=0.1, dt=0.001, normalized=0.0)
            I_paths[:, t_idx] = first.(sol.u)
        end
        mean_path_full = mean(I_paths, dims=2)[:]
        # ---------------------------------------------

        mean_path_zoom = mean_path_full[zoom_indices]
        
        if row.method == "SSE"
            plot!(p_zoom, zoom_time, mean_path_zoom, lw=2, color=:red, label="")
            continue
        end

        S_val, M_val = extract_MS(row.method)
        color = color_from_S(S_val)
        marker = marker_map[M_val]

        # Ploteo de la línea de zoom (sin leyenda)
        plot!(p_zoom, zoom_time, mean_path_zoom,
              lw=2, color=color,
              marker=marker, markersize = 3, label="")
        
    end
    # --- FIN LOOP DE ZOOM ---

    # 3. CREAR EL PLOT PARA LA BARRA DE COLOR (LEYENDA SUPERIOR)
    # ... (código de p_cbar) ...

    S_palette = [color_from_S(s) for s in LinRange(minS, maxS, 100)];

    S_data_range = collect(LinRange(minS, maxS, 100))


    # Creamos un plot ficticio de 1x1 para sostener el colorbar
    heatmap!(p_zoom, reshape(S_data_range, 100, 1), color=S_palette, alpha=0.0, colorbar=true, cbar_title="S")

    # 4. COMBINAR GRÁFICOS (SOLUCIÓN ESTRUCTURAL DEFINITIVA)
     

    # Definimos el layout usando un macro que nos permita especificar las proporciones
    layout_matrix = @layout [a ; b c]

    p_final = plot(        # a
        p_main,          # b
        p_zoom,
        size = (1400, 600)
    )

    display(p_final)
end
    
plot_comparison_with_zoom(final_summary)

simulate_sis_sde(θ0[1], γ, μ, θ0[2], [I[1]*N], N, 10, saveat = 0.1, dt = 0.001, normalized = 0.0).u

I_paths = zeros(nSims, length(time_points))
I_paths[1, :]


summary_df = combine(
    groupby(smle_df, [:S, :M]),
    :β => mean => :Mean_beta,
    :σ => mean => :Mean_sigma,
    :β => std => :Std_beta,
    :σ => std => :Std_sigma,
    :β => (x -> quantile(x, 0.975) - quantile(x, 0.025)) => :CI_Length_beta,
    :σ => (x -> quantile(x, 0.975) - quantile(x, 0.025)) => :CI_Length_sigma,
)
    
M_values = sort(unique(summary_df.M))
    
# Asignación de marcadores y colores (simples)
marker_list = [:circle, :square, :diamond, :utriangle]
marker_map = Dict(M => marker_list[i] for (i, M) in enumerate(M_values))
    
# Inicializar el gráfico
p = plot(
    title = "Convergencia del Estimador Beta (β) vs. Simulaciones (S)",
    xlabel = "S (Número de Simulaciones Monte Carlo)",
    ylabel = "Estimador Promedio de Beta (Mean β)",
    legend = :right,
    xscale = :log10, # Usar escala logarítmica para S es común
    size = (800, 600)
)

# Plotear cada serie de M
for M in M_values
    # Filtrar el DataFrame para obtener la serie M actual
    df_M = filter(row -> row.M == M, summary_df)
        
    # Plotear la serie
    plot!(p, df_M.S, df_M.Mean_beta, 
        label = "M = $(M)",
        marker = marker_map[M],
        markersize = 5,
        lw = 1.5
    )
end

function plot_convergence_precision(summary_df::DataFrame, param::Symbol, metric::Symbol)
    
    M_values = sort(unique(summary_df.M))
    marker_list = [:circle, :square, :diamond, :utriangle]
    marker_map = Dict(M => marker_list[i] for (i, M) in enumerate(M_values))
    
    y_label = metric == :Std_beta ? "Desviación Estándar de β (Std β)" : 
              metric == :CI_Length_beta ? "Longitud CI 95% de β" : 
              metric == :Std_sigma ? "Desviación Estándar de σ (Std σ)" : 
              "Métrica de Precisión"

    p = plot(
        title = "Precisión de $(string(param)) vs. Simulaciones (S)",
        xlabel = "S (Número de Simulaciones Monte Carlo)",
        ylabel = y_label,
        legend = :right,
        xscale = :log10,
        size = (800, 600),
        minorgrid = true # Para mejor visualización en escala log
    )

    # Plotear cada serie de M
    for M in M_values
        df_M = filter(row -> row.M == M, summary_df)
        
        plot!(p, df_M.S, df_M[!, metric], 
            label = "M = $(M)",
            marker = marker_map[M],
            markersize = 5,
            lw = 1.5
        )
    end
    
    display(p)
end

plot_convergence_precision(summary_df, :beta, :Std_beta)
    
