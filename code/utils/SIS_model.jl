using DifferentialEquations
using Interpolations 
using QuadGK
using Plots

default_normalized = false

function sis_sde!(du, u, p, t)
    du .= f(u, p, t)
end


function f(u, p, t)
    I = u[1]
    β, γ, μ, σ, N, normalized = p
    if normalized == 1.0
        β = N*β
        σ = N*σ
        N = 1.0
    end
    du = [β * I * (N - I) - (μ + γ) * I]
    return du
end


function sis_noise!(du, u, p, t)
    du .= g(u, p, t)
end


function g(u, p, t)
    I = u[1]
    β, γ, μ, σ, N, normalized = p
    if normalized == 1.0
        β = N*β
        σ = N*σ
        N = 1.0
    end
    du = [σ * I * (N - I)]
    return du
end


function simulate_sis_sde(β, γ, μ, σ, X0, N, tf; normalized = 0.0, saveat = 1.0, dt = 0.01)

    @assert tf > 0 "El tiempo final debe ser positivo."
    u0 = X0
    tspan = (0.0, tf)
    p = [β, γ, μ, σ, N, normalized]

    prob = SDEProblem(
                    (du, u, p, t) -> sis_sde!(du, u, p, t), 
                    (du, u, p, t) -> sis_noise!(du, u, p, t), 
                    u0, tspan, p)
    sol = solve(prob, EM(), dt=dt, saveat=saveat)
    #I_vals = sol[2,1:end]  # número de infectados
    return sol
end

# MODEL A
sim = simulate_sis_sde(2.55504e-6, 1/55, 1/(40*365.25), 1e-6, [1000], 10_000, 6000, saveat = 0.1, dt = 0.001)
plot(sim)

# MODEL B
sim = simulate_sis_sde(2.8650e-7, 0.02011, 1.3736e-3, 1e-6, [1000], 150_000, 6000, saveat = 0.1, dt = 0.001)
plot(sim)

sim = simulate_sis_sde(8.753e-9, 1/(21/365.25), 1/(82.67-15), 1.654537e-8, [112.3450], 39591836, 12, saveat = 1, dt = 0.001)



plot(sim)

function simulate_many(β, γ, σ, s, X0, N, tf; n_sims=1000, saveat = 1.0, normalized = true, dt = 0.01)
    # correr una simulación para saber longitud
    nT = 0
    I_vals = false
    ts = []
    while true
        try
            I_vals = simulate_sir_sde(β, γ, σ, s, X0, N, tf, saveat = saveat, normalized = normalized, dt = dt)
            ts = I_vals.t
            I_vals = I_vals[2, :]
            nT = length(I_vals)
            break
        catch 
        end 
    end
    # matriz para guardar resultados
    I_all = zeros(n_sims, nT)
    I_all[1, :] .= I_vals
    # correr simulaciones restantes
    for j in 2:n_sims
        while true
            try
                I_all[j, :] .= simulate_sir_sde(β, γ, σ, s, X0, N, tf, saveat = saveat, normalized = normalized, dt = dt)[2, :]
                break
            catch 
            
            end
        end
    end
    # media por tiempo
    I_mean = mean(I_all, dims=1) |> vec
    return I_mean, ts, I_all
end

"""
    Obtiene un estimado de las sigmas usando una observación del proceso,
    suponiendo que son observaciones continuas.
"""
function sigmas_by_qv(S, I, t; normalized = default_normalized)
    qvS = sum(diff(S).^2)
    qvI = sum(diff(I).^2)

    ex = normalized ? 1 : 2
    cte = normalized ? N : 1

    LS = linear_interpolation(t, S)
    LI = linear_interpolation(t, I)

    Integral1, _ = quadgk(t -> (LS(t)*LI(t)/cte)^ex, t[1], t[end])
    Integral2, _ = quadgk(t -> (LI(t))^ex, t[1], t[end])

    σβ = sqrt(qvS / Integral1)
    σγ = sqrt((qvI - qvS) / Integral2)

    return σβ, σγ
end

