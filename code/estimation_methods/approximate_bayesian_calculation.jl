using Plots
using DifferentialEquations
using Distributions
using QuadGK
using Interpolations
using Statistics
using ProgressMeter
using Base.Threads
println("Número de hilos: ", nthreads())

script_dir = @__DIR__  
include(joinpath(script_dir, "..", "utils/data_getters.jl"))
include(joinpath(script_dir, "..", "utils/SIR_model.jl"))
include(joinpath(script_dir, "..", "utils/functions.jl"))

normalized = false

# si supongo que duracion = 10 dias, entonces 10/360 años, 

dt = 1/360
function reconstruir_obs(gamma_est, sigma_gamma_est)
    dt = 1/360
    df = get_new_infections_COVID_by_state("Ciudad de México")
    df = filter(:fecha => f -> Date(2020,3,8) ≤ f ≤ Date(2020, 4, 26), df)
    # df.casos = rollmean(df.casos, 6)
    N = get_population_ent("Ciudad de México")

    R = zeros(length(df.casos))
    I = zeros(length(df.casos))

    I[1] = df.casos[1]

    for t in 2:length(df.casos)
        dW = rand(Normal(0, sqrt(dt)))
        I[t] = I[t-1] + df.casos[t] - (1/gamma_est)*I[t-1]*dt - sigma_gamma_est*I[t-1]*dW
        R[t] = R[t-1]  + (1/gamma_est)*I[t-1]*dt + sigma_gamma_est*I[t-1]*dW
    end

    return N .- (R.+I), I, R
end

S, I, R = reconstruir_obs(10, 0.05)

plot(I)


plot(df.I, label ="reales I")
plot(df.S, label ="reales S")

plot(sol[2, :], label ="synth I")
plot(sol[1, :], label ="synth S")

obs = hcat(S, I)
obs_ts = collect(0:1:(length(I)-1))

obs_ts *= 1/360

sb, sg = sigmas_by_qv(S, I, obs_ts, normalized = normalized)


#sb, sg = sigmas_by_qv(obsS, obsI, obsts, normalized = normalized)

using GLM

function est_beta(S, I, t, N, normalized = normalized)
    Y = -diff(S) ./ (I[1:end-1])
    X = S[1:end-1] / N
    data = DataFrame(X = X, Y = Y)
    ols = lm(@formula(Y ~ X + 0), data)
    beta_est = GLM.coef(ols)[1]
    se_beta = GLM.stderror(ols)[1]
    return beta_est, se_beta
end

N = 763

b, sd = est_beta(df.S, df.I, obs_ts, N)
best = mean(b)

histogram(b)

# Guardar a CSV
CSV.write("SIRCDMX.csv", df)



xβ  = range(minimum(b),  maximum(b),  length=200)
distB = fit(Normal, b)
plot(xβ, pdf.(distB, xβ))
histogram!(b, normalize = true)

function abc_rejection_sir_sde_matrix(obs;
                                      #prior_β = Uniform(maximum([0, minimum(b)]), maximum(b)),
                                      prior_β = Truncated(distB, o, Inf),
                                      prior_γ = Uniform(2.0, 7.0),
                                      prior_σβ = Truncated(Normal(sb,2*sb), 0, Inf),
                                      prior_σγ = Truncated(Normal(sg, 2*sg), 0, Inf),
                                      n_sims = 100_000,
                                      epsilon = 30.0,
                                      dist = dist_l2,
                                      normalized = false,
                                      tf = length(df.I)-1)

                                     
    p = Progress(n_sims)  # n_sims pasos, actualización cada 1 iteración

    accepted_params = Float64[]  # inicial vacío

    for i in 1:n_sims
        β = rand(prior_β)
        γ = 1/rand(prior_γ)  # cuidado, inviertes aquí
        σβ = rand(prior_σβ)
        σγ = rand(prior_σγ)
        X0 = obs[1, :]
        #I0 = rand(i0)
        try
            sim = simulate_sir_sde(β, γ, σβ, σγ, X0, N, tf, normalized = normalized)
            d = dist(obs, sim)
            if d ≤ epsilon
                # agregar fila (β, γ, σ)
                append!(accepted_params, [β, γ, σβ, σγ])
            end
        catch
        end
        next!(p)
    end

    # convertir en matriz de 3 columnas: cada fila = un conjunto de parámetros
    n_accepted = length(accepted_params) ÷ 4
    return reshape(accepted_params, 4, n_accepted)'  # filas = parámetros, columnas = β,γ,σ
end

using Base.Threads

function abc_rejection_sir_sde_matrix_parallel(obs, X0;
                                      prior_β =  Truncated(Normal(b, sd), 0, Inf),
                                      prior_γ = Uniform(7.0, 21.0),
                                      prior_σβ = Truncated(Normal(b, 3*b), 0, Inf),
                                      prior_σγ = Truncated(Normal(1/10, 3/10), 0, Inf),
                                      n_sims = 100_000,
                                      epsilon = 30.0,
                                      dist = dist_l2,
                                      normalized = false,
                                      tf = length(df.I)-1)

    # Contenedor seguro por hilos
    accepted_channel = Channel{Vector{Float64}}(n_sims) 
    p = Progress(N, desc="Calculando con Threads: ")

    Threads.@threads for i in 1:n_sims
        β = rand(prior_β)
        γ = 1/rand(prior_γ)
        σβ = rand(prior_σβ)
        σγ = rand(prior_σγ)
        tmp = Float64[]
        try
            sim = simulate_sir_sde(β, γ, σβ, σγ, X0, N, tf, normalized = normalized)
            d = dist(obs, sim)
            if d ≤ epsilon
                put!(accepted_channel, [β, γ, σβ, σγ]) 
            end
        catch 
            # ignorar errores
        end
        next!(p)
    end

    close(accepted_channel)
    accepted_list = collect(accepted_channel)

    accepted_params = reduce(vcat, accepted_list; init = Float64[])
    n_accepted = length(accepted_params) ÷ 4
    
    # Si no se aceptó ninguno, devuelve una matriz vacía
    if n_accepted == 0
        return zeros(0, 4)
    end

    return reshape(accepted_params, 4, n_accepted)'

end


obsS = linear_interpolation(sol.t, sol[1, :])
obsI = linear_interpolation(sol.t, sol[2, :])

obsI(30)
plot(sol[2, :])

function dist_l2(obs, sim)
    return (abs(obs[20, 2]) .- sim[2, 20])+ abs(obs[50, 2] .- sim[2, 50])
end 
# con 100M de simulaciones tardó 2375.526328 segundos
# obtuvo 130 muestras
@time res = abc_rejection_sir_sde_matrix_parallel(obs, obs[1, :], epsilon = 200.0, n_sims = 100_000, 
            normalized = normalized, tf = 50)

bb = median(res[:, 1])
gg = median(res[:, 2])
ssb = median(res[:, 3])
ssg = median(res[:, 4])

θ = (bb, gg, ssb, ssg)

sim, ts, _ = simulate_many(bb, gg, ssb, ssg, [obsS(1), obsI(1)], N, 50, 
            normalized = normalized, saveat = 0.01);
plot(ts, sim, label = "simulacion")
plot!(sol.t, sol[2, :], label = "datos reales")

#Iobs|Ireal ∼ Beta(α, β)

# Convertir a DataFrame
df = DataFrame(res, :auto)  # :auto nombra columnas automáticamente

# Guardar a CSV
CSV.write("ABC100M.csv", df)

CSV.read("ABC100M.csv", DataFrame())
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #

# === Posteriors (de tus resultados ABC) ===
β_post  = res[:, 1]
γ_post  = res[:, 2]
σβ_post = res[:, 3]
σγ_post = res[:, 4]

# === Rango de valores para graficar ===
xβ  = range(minimum(vcat(β_post, b)),  maximum(vcat(β_post, b)),  length=200)
xγ  = range(minimum(γ_post),  maximum(γ_post),  length=200)
xσβ = range(minimum(σβ_post), maximum(σβ_post), length=200)
xσγ = range(minimum(σγ_post), maximum(σγ_post), length=200)

p1 = plot_posterior_distribution(β_post, interval = xβ, prior = distB)
p2 = plot_posterior_distribution(γ_post, interval = xγ, prior = Uniform(7.0, 21.0))
p3 = plot_posterior_distribution(σβ_post, interval = xσβ, prior = Truncated(Normal(sb, 2sb), 0, Inf))
p4 = plot_posterior_distribution(σγ_post, interval = xσγ, prior = Truncated(Normal(sg, 2sg), 0, Inf));

# === Mostrar en grid 2x2 ===
plot(p1, p2, p3, p4, layout=(2,2), size=(800,600))
