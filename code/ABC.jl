using RCall
using Plots
using DifferentialEquations
using Distributions
using QuadGK
using Interpolations
using Statistics

function sir_sde!(du, u, p, t; normalized = true)
    S, I, R = u
    β, γ, σ, N = p
    cte = normalized ? 1 : N
    du[1] = -β*S*I/cte
    du[2] = β*S*I/cte - γ*I
    du[3] = γ*I
end

function sir_noise!(du, u, p, t; normalized = true)
    S, I, R = u
    β, γ, σ, N = p
    #exp, exp2 = normalized ? (1/2, 1/2) : 1,0
    du[1] = (-(σ*S*I)^1) / 1
    du[2] = ((σ*S*I)^1) / 1
    du[3] = 0.0
end

function simulate_sir_sde(β, γ, σ, I0, N, tf; saveat = 1.0, normalized = true, dt = 0.01)
    cte = normalized ? 1 : N
    S0 = cte - I0
    R0 = 0.0
    @assert I0 > 0 "El valor inicial debe ser positivo."
    @assert cte == S0 + I0 + R0 "El valor inicial no es válido: S₀ + I₀ + R₀ != $cte"
    @assert tf > 0 "El tiempo final debe ser positivo."
    u0 = [S0, I0, R0]
    tspan = (0.0, tf-1)
    p = [β, γ, σ, N]
    prob = SDEProblem(
                    (du, u, p, t) -> sir_sde!(du, u, p, t, normalized = normalized), 
                    (du, u, p, t) -> sir_noise!(du, u, p, t, normalized = normalized), 
                    u0, tspan, p)
    sol = solve(prob, EM(), dt=dt, saveat=saveat)
    I_vals = sol[2,1:end]  # número de infectados
    return I_vals
end

function simulate_many(β, γ, σ, I0, N, tf; n_sims=1000, saveat = 1.0, normalized = true, dt = 0.01)
    # correr una simulación para saber longitud
    I_vals = simulate_sir_sde(β, γ, σ, I0, N, tf, saveat = saveat, normalized = normalized, dt = dt)
    nT = length(I_vals)
    # matriz para guardar resultados
    I_all = zeros(n_sims, nT)
    I_all[1, :] .= I_vals
    # correr simulaciones restantes
    for j in 2:n_sims
        I_all[j, :] .= simulate_sir_sde(β, γ, σ, I0, N, tf, saveat = saveat, normalized = normalized, dt = dt)
    end
    # media por tiempo
    I_mean = mean(I_all, dims=1) |> vec
    return I_mean, I_all
end


I = [25, 75, 227, 296, 258, 236, 192, 126, 71, 28, 11, 7]

plot(I)
N = 763

normalized = false

# distancia L2 (Euclídea) entre vectores

#dist_l2(sim, obs) = sum(abs(sim .- obs)*N)/length(sim)

dist_l2(sim, obs) = (abs(sim[3]-obs[3]) + abs(sim[6]-obs[6]))

# I1 = I0 + bI0/N(N-I0) - gamma I0
# I1 - I0 +
I = I/N
b0 = (I[2]-I[1] + (1/7)*I[1])*N/(I[1]*(N-I[1]))

# I1 - I0 = bI0S0 - γI0


function SIR_sigma_by_qv(S, I, ts, N)
    St = linear_interpolation(ts, S)
    It = linear_interpolation(ts, I)
    sqrt( sum( diff(I).^2 ) / quadgk(s -> ( (It(s)*St(s))^2 ) / N, ts[1], ts[end])[1])
end

ss = SIS_sigma_by_qv(I, 2:1:13)

tf = 13


function abc_rejection_sir_sde_matrix(obs;
                                      prior_β = Uniform(0.0, 3.0),
                                      prior_γ = Uniform(1.0, 10.0),
                                      prior_σ = Uniform(0.0, ss*2),
                                      i0 = I[1],
                                      n_sims = 100_000,
                                      epsilon = 30.0,
                                      dist = dist_l2,
                                      normalized = true)

    accepted_params = Float64[]  # inicial vacío

    for i in 1:n_sims
        β = rand(prior_β)
        γ = 1/rand(prior_γ)  # cuidado, inviertes aquí
        σ = rand(prior_σ)
        I0 = i0
        #I0 = rand(i0)
        try
            sim = simulate_sir_sde(β, γ, σ, I0, N, tf, normalized = normalized)
            d = dist(sim, obs)
            if d ≤ epsilon
                # agregar fila (β, γ, σ)
                append!(accepted_params, [β, γ, σ, I0])
            end
        catch
        end
    end

    # convertir en matriz de 3 columnas: cada fila = un conjunto de parámetros
    n_accepted = length(accepted_params) ÷ 4
    return reshape(accepted_params, 4, n_accepted)'  # filas = parámetros, columnas = β,γ,σ
end


res = abc_rejection_sir_sde_matrix(I, epsilon = 20.0, n_sims = 10_000, normalized = normalized)

b = mean(res[:, 1])
g = mean(res[:, 2])
s = mean(res[:, 3])
i0 = mean(res[:, 4])

sim, _ = simulate_many(b,g,s, i0, N, tf, saveat = 1, normalized = false)
plot(sim, label = "simulacion")
plot!(I, label = "datos reales")

# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #

# carga el archivo .RData en el entorno de R
R"load('data.Rdata')"
R"C <- cdmx.acumulados"
R"I <- cdmx.acum14"
@rget I
@rget C
I = I[20:end]
C = C[20:end]
S = N .- C
plot(S)
# 10 de marzo al 25 de abril
plot(I)
I = I/N
S = S/N
plot(I, label = "datos reales CDMX")
N = 9_018_645
N = 963861

function SIR_sigma_by_qv(S, I, ts, N)
    sqrt(sum( diff(I).^2 ) / sum((S .* I ./N).^2))
end

ss = SIR_sigma_by_qv(S, I, 1:1:47, N)

dist_l2(sim, obs) = sum(abs(sim.- obs).*N)/length(sim)
dist_l2(sim, obs) = (abs(sim[40]-obs[40])+abs(sim[20]-obs[20]))

function dist_l2(sim, obs; val = 100)
    indicador(x) = x < val ? 0 : 1

    return sum(indicador(abs(sim[25]-obs[25])) + 
               indicador(abs(sim[50]-obs[50])) +
               indicador(abs(sim[75]-obs[75])))
end

normalized = false
I = I/N
b0 = (I[2]-I[1]+(1/7)*I[1])*N/(I[1]*(N-I[1]))

S7 = 1 .- I[1:7]
I7 = I[1:7]
b0 = sum((S[1:46].*I[1:46]/N).*(diff(I[1:47]).+(1/7).*I[1:46]))/sum((S[1:46].*I[1:46]/N).^2)

tf = 47
I = I[1:47]

explosiones = 0

b = b0
s = ss

res = abc_rejection_sir_sde_matrix(I, 
                                   prior_β = Normal(b, 1.0),
                                   prior_γ = Uniform(7.0, 21.0),
                                   prior_σ = Normal(s, 0.01),
                                   i0 = I[1],
                                   epsilon = 100.0, 
                                   n_sims = 10_000,
                                   normalized = normalized)

plot(simulate_sir_sde(b, 1/7, s, I[2], N, tf, normalized = true))
plot!(I)

using StatsBase

b = mode(res[:, 1])
g = mode(res[:, 2])
s = mode(res[:, 3])
i0 = mode(res[:, 4])

tf = 47
#bbueno, gbueno, sbueno, ibueno = b,g,s,i0
sim, _ = simulate_many(b,g,s,i0,N,tf,n_sims = 10, saveat = 0.001)
plot(collect(range(0.0, stop = tf, length = length(sim))), sim, label = "simulacion")
plot!(I[1:tf], label = "datos reales")


