using DifferentialEquations
using Interpolations 
using QuadGK

default_normalized = false

function sir_sde!(du, u, p, t; normalized = default_normalized)
    du .= f(u, p, t, normalized = normalized)
end

function f(u, p, t; normalized = default_normalized)
    S, I = u
    β, γ, σ, s, N = p
    cte = normalized ? 1 : N
    du = [-β*S*I/cte,
    β*S*I/cte - γ*I]
    return du
end

"""
Σ = (
    -√(σ₁SI/N), 0.0
    √(σ₁SI/N), -√(σ₂I)
)   
"""
function sir_noise!(du, u, p, t; normalized = default_normalized)
    du .= g(u, p, t, normalized = normalized)
end

function g(u, p, t; normalized = default_normalized)
    S,I = u
    β, γ, σ,s, N = p

    cte = normalized ? N : 1
    ex = normalized ? (1/2) : 1

    du = zeros(2,2)

    # llenar matriz de difusión 2x2
    du[1,1] =  -σ*(S*I / cte)^ex  ; du[1,2] = 0.0;
    du[2,1] =  σ*(S*I / cte)^ex   ; du[2,2] = -s*(I)^ex;
    return du
end

"""function simulate_sir_sde(β, γ, σ, I0, N, tf; saveat = 1.0, normalized = true, dt = 0.01)
    cte = normalized ? 1 : N
    S0 = cte - I0
    R0 = 0.0
    @assert I0 > 0 "El valor inicial debe ser positivo."
    @assert cte == S0 + I0 + R0 "El valor inicial no es válido: S₀ + I₀ + R₀ != cte"
    @assert tf > 0 "El tiempo final debe ser positivo."
    u0 = [S0, I0, R0]
    tspan = (0.0, tf-1)
    p = [β, γ, σ, N]
    prob = SDEProblem(
                    (du, u, p, t) -> sir_sde!(du, u, p, t, normalized = normalized), 
                    (du, u, p, t) -> sir_noise!(du, u, p, t, normalized = normalized), 
                    u0, tspan, p)
    sol = solve(prob, EM(), dt=dt, saveat=saveat)
    #I_vals = sol[2,1:end]  # número de infectados
    return sol
end"""

function simulate_sir_sde(β, γ, σ, s, X0, N, tf; saveat = 1.0, normalized = true, dt = 0.01)
    cte = normalized ? 1 : N
    @assert tf > 0 "El tiempo final debe ser positivo."
    u0 = X0
    tspan = (0.0, tf)
    p = [β, γ, σ, s, N]
    noise = ones(2,2)

    prob = SDEProblem(
                    (du, u, p, t) -> sir_sde!(du, u, p, t, normalized = normalized), 
                    (du, u, p, t) -> sir_noise!(du, u, p, t, normalized = normalized), 
                    u0, tspan, p, noise_rate_prototype = noise)
    sol = solve(prob, EM(), dt=dt, saveat=saveat)
    #I_vals = sol[2,1:end]  # número de infectados
    return sol
end

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

