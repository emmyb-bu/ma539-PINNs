using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters t, x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C=1
eq  = Dtt(u(t,x)) ~ C^2*Dxx(u(t,x))

u₀(x) = exp(-(x - 1/2)^2/(2*0.04^2))

# Initial and boundary conditions
bcs = [u(t,0) ~ 0.,# for all t > 0
       u(t,1) ~ 0.,# for all t > 0
       u(0,x) ~ u₀(x), #for all 0 < x < 1
    #    Dt(u(0,x)) ~ 0. 
       ] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]
# Discretization
dx = 0.005;
dt = 0.005;

# Neural network
chain = Lux.Chain(Dense(2,16,Lux.σ),Dense(16,16,Lux.σ),Dense(16,1))
discretization = PhysicsInformedNN(chain, GridTraining([dx,dt]),adaptive_loss = NonAdaptiveLoss(pde_loss_weights=1.0,bc_loss_weights=7))

@named pde_system = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
prob = discretize(pde_system,discretization)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end

# optimizer
opt = OptimizationOptimJL.BFGS()
res = Optimization.solve(prob,opt; callback = callback, maxiters=1500)
phi = discretization.phi


using JLD2
# res = jldopen("wave_eq_sol-X-bcs.jld2")
# res = jldopen("wave_eq_sol-nice.jld2")

using Plots, PlotThemes, LaTeXStrings; theme(:dao)

ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]

u_predict = reshape([first(phi([t,x],res.u)) for x in xs for t in ts],(length(ts),length(xs)))
plot(xs,ts, u_predict, linetype=:contourf,title = L"u(t,x)",xlabel=L"x",ylabel=L"t")
# savefig("nice-wave-NeuralPDE.pdf")

plot()
for t in 0:0.1:1
    plot!(xs,[first(phi([t,x],res.u)) for x in xs],ylim=(-1.1,1.1),label="t = $t",linecolor=ColorSchemes.matter[201-(Int∘floor)(200*t)])
end
plot!(xlabel=L"x",ylabel=L"u(t,x)")
# savefig("nice-wave-NeuralPDE-per-t.pdf")
