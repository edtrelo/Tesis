using Distributions
using BlackBoxOptim
script_dir = @__DIR__  
include(joinpath(script_dir, "..", "utils/data_getters.jl"))
include(joinpath(script_dir, "..", "utils/SIR_model.jl"))

# k: número de variables del proceso
# μ: R^k x R+ -> R^k Función de deriva
# Σ: R^k x R+ -> R^k x R^k
# θ es L dimensional 

# Función que aproxima la densidad de transisión p(Y_tn+1, tn+1| Y_tn, tn; θ)
# Se elige la discretización de Euler en M subintervalos de longuitud (tn+1-tn)/M 

M = 100
S = 1000
using LinearAlgebra

function q(θ, M, S, Ynext, tnext, Ycurrent, tcurrent, N; normalized = false)
    h = (tnext - tcurrent)/M
    β, γ, σ1, σ2 = θ
    suma = 0
    # S es el número de muestras MC 
    for s in 1:S 
        # Z es una variable aleatoria 
        # Su distribución es f(z) = q_M(z, t_n + (M-1)h| Y_tn, tn)
        Z = simulate_sir_sde(β, γ, σ1, σ2, Ycurrent, N, (tnext - tcurrent), 
            dt = h, normalized = normalized, saveat = h)
        # Tomamos el último valor de Z 
        zs = Z[end-1]
        Σ = g(zs, vcat(θ, N), 0)*g(zs,vcat(θ, N),0)' *h
        #println(Σ)
        #Σ = (Σ + Σ') / 2  # forzar simetría
        #Σ += 1e-8*I
        #println(Σ)
        Dist = MvNormal(zs .+ f(zs, vcat(θ,N), 0)*h,  Σ)
        suma += pdf(Dist, Ynext)
    end 
    return suma/S
end

g([0.9, 0.05, 0.05], [0.9, 0.02, 0.1, 0.1, 1], 0)*g([0.9, 0.05, 0.05], [0.9, 0.02, 0.1, 0.1, 1], 0)'

function verosimilitud(x_0)
    try 
        suma = 0
        if sum(x_0 .<= 0) > 0
            return -Inf
        end
        for i in 2:(length(obs)-1)
            #dz = DZ[i]
            suma += log(q(x_0, M, S, obs[i, :], obs_ts[i], obs[i-1, :], obs_ts[i-1], N))
        end
        return -suma
    catch  
        return -1e-12
    end
end

normalized = false

df = get_new_infections_COVID_by_state("Ciudad de México")
df = filter(:fecha => f -> Date(2020,3,8) ≤ f ≤ Date(2020, 4, 26), df)
N = get_population_ent("Ciudad de México")

df.I = rolling_sum_last10(df.casos) / (normalized ? N : 1)
plot(df.I)
df.C = cumsum(df.casos) / (normalized ? N : 1)
df.S = (normalized ? 1 : N) .- df.C
plot(df.S)
df.R = (normalized ? 1 : N) .- (df.S .+ df.I)
df = filter(:I => i -> i > 0, df)

obs = hcat(df.S, df.I, df.R)

obs_ts = collect(0:1:49)


# optimización con Fminbox
res = bboptimize(
    verosimilitud,
    SearchRange = [(1e-8, 5.), (1e-8, 2.), (1e-8, 5.), (1e-8, 5.)],
    Method = :adaptive_de_rand_1_bin_radiuslimited,   # Differential Evolution
    MaxSteps = 1_000,
    PopulationSize = 50
)
    

sol = best_candidate(res)

sim = simulate_sir_sde(sol[1], sol[2], sol[3], sol[4], obs[1,:], N,
        50, saveat=0.01, normalized = normalized, dt = 0.001)

I_mean, I_all = simulate_many(sol[1], sol[2], sol[3], sol[4], obs[1,:], N,
        50, n_sims=1000, saveat = 0.01, normalized = false, dt = 1e-3)

plot(sim.t, I_mean, label = "simulacion")
plot!(df.I, label = "casos reales")