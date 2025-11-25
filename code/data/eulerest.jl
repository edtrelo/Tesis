using CSV
using DataFrames
using Statistics
using Optim
using Plots

# --- Leer datos ---
df = CSV.read("code/data/gonorrea_casos_espana.csv", DataFrame)

df[!, "prevalence"] = df[!, "gonorhea_incidence"].*(21/360)

# Calcular prevalencia y proporciones
Iobs = df.prevalence ./ 39591836    # proporción infectada
tobs = collect(0:(length(Iobs)-1)) 

N = 1.0              # trabajamos en proporciones
N = 39591836
μ = 1 / 82.67
γ = 360 / 21

# --- Graficar prevalencia ---
plot(tobs, Iobs, xlabel="Tiempo", ylabel="Proporción infectada", title="Prevalencia observada")

# --- Integral por regla del trapecio ---
function trapz(x, y)
    sum(diff(x) .* ((y[1:end-1] .+ y[2:end]) ./ 2))
end

integral = trapz(tobs, (Iobs .* (N .- Iobs)).^2)
σ = sqrt(sum(diff(Iobs).^2) / integral)

# --- Función de log-verosimilitud ---
function likelihood_sis(beta)
    n = length(Iobs)
    dt = diff(tobs)
    Iold = Iobs[1:end-1]
    Sold = N .- Iold
    Ifuture = Iobs[2:end]
    
    term1 = -(n/2)*log(2π)
    term2 = -0.5 * sum(
        log.(σ^2 .* Iold.^2 .* Sold.^2 .* dt) .+
        ((Ifuture .- Iold .- dt .* (beta .* Iold .* Sold .- (μ + γ) .* Iold)).^2) ./ 
        (σ^2 .* Iold.^2 .* Sold.^2 .* dt)
    )
    return term1 + term2   # devuelve log-verosimilitud (no negativa)
end

println("Log-verosimilitud en β=3.33: ", likelihood_sis(3.33, σ))

# --- Estimación por máxima verosimilitud ---
obj(params) = -likelihood_sis(params[1])  # se minimiza la negativa

res = optimize(obj, [1e-6], [100.0], [10.5], Fminbox(LBFGS()))

β_est = Optim.minimizer(res)[1]
σ_est = β_est = Optim.minimizer(res)[2]
logLik = -Optim.minimum(res)

println("\nEstimación de máxima verosimilitud:")
println("β̂ = ", β_est)
println("Log-verosimilitud = ", logLik)

using DifferentialEquations, Plots


I_0 = [Iobs[1]]
tmax = 10.0

# --- Definición del drift (determinístico) ---
function f!(dI, I, p, t)
    β, μ, γ, N = p
    dI[1] = β * I[1] * (N - I[1]) - (μ + γ) * I[1]
end

# --- Definición del término de difusión (estocástico) ---
function g!(dI, I, p, t)
    σ, N = p[1], p[4]
    dI[1] = σ * I[1] * (N - I[1])
end

# --- Condición inicial y parámetros ---
p = (β_est, μ, γ, N, σ)
p = (2.0055e−6, 1.3736e-3, 0.02011, 150_000, 1e-6)

using DifferentialEquations

# --- Definir el problema estocástico ---
prob = SDEProblem(f!, g!, [50000], (0.0, 3000), p)

# --- Resolver con método de Euler–Maruyama ---
sol = solve(prob, RKMil(), dt=0.001)

# --- Graficar ---
plot(sol, lw=2, xlabel="Tiempo", ylabel="Proporción infectada",
     title="Simulación del modelo SIS estocástico", legend=false)

