using ForwardDiff, NNlib, JLD2
const FD = ForwardDiff;

numType = Float32;

# Define a simple feed-forward neural network:

nnLayer(x,W,b,σ) = σ.(reshape(W,(size(b,1),:))*x .+ b)

function getRandomParams(layerStructure,σ)
    """
    Given a layer structure, generate random neural network parameters
    """
    nParams = sum(X[1][1]*X[1][2]+X[1][2] for X in layerStructure);
    return σ*randn(numType,nParams)
end

function neuralNet(x,layerStructure::Vector{Tuple{Pair{Int64, Int64}, Function}},params)
    """
    Compile a layer structure and evaluate on data point(s)
    """
    X = x;
    n = 1;
    for (layer, σ) in layerStructure
        X = nnLayer(X,params[n:(n-1+layer[1]*layer[2])],params[(n+layer[1]*layer[2]) : (n-1+layer[1]*layer[2]+layer[2])],σ)
        n += layer[1]*layer[2]+layer[2];
    end
    return X
end

layerStructure = [
    (2 => 18,NNlib.σ),
    (18 => 18,NNlib.σ),
    (18 => 1,identity)
];

# Initialize a network

ϕmodel(xt,params) = neuralNet(xt,layerStructure,params);
initParams = getRandomParams(layerStructure,1.0);

@info "Neural network intialized"

xmin = 0; xmax = 1; Tmax = 1;
V(x) = 0;
# nCollocationPts = 500;
nxCollocationPts = 20;
nyCollocationPts = 20;
nCollocationPts = nxCollocationPts * nyCollocationPts;
n∂t0ptscenter = 50;
n∂t0ptsedges = 30;
n∂t0pts=n∂t0ptscenter+n∂t0ptsedges;
n∂xpts = 50;
collocationPts = hcat(reshape(collect.(Iterators.product(range(0,xmax,nxCollocationPts), range(0,xmax,nyCollocationPts))),:)...) |> Matrix{numType};
∂t0pts = hcat(hcat(reshape(collect.(Iterators.product(range(0,xmax,n∂t0ptsedges), 0:0)),:)...),hcat(reshape(collect.(Iterators.product(range(xmax/2 - 0.1,xmax/2 + 0.1,n∂t0ptscenter), 0:0)),:)...)) |> Matrix{numType}
∂xpts = hcat(reshape(collect.(Iterators.product(0:1, range(0,Tmax,n∂xpts))),:)...)  |> Matrix{numType};
σinit = 0.05;
ϕt0 = (xt -> exp(-(xt[1] - xmax/2)^4/(2*σinit^4))).(eachcol(∂t0pts)) |> Vector{numType};
dϕt0 = (xt -> 0).(eachcol(∂t0pts)) |> Vector{numType};
ϕx0 = zeros(2n∂xpts) |> Vector{numType};


# Define loss function:

function diffusionLoss(model,params)
    Dfs = [FD.gradient(Y->model(Y,params)[1],X) for X in eachcol(collocationPts)];
    DDfs = [FD.jacobian(X -> FD.gradient(Y->model(Y,params)[1],X),Z) for Z in eachcol(collocationPts)];
    return (
        # PDE loss
        NNlib.mean(abs2(Dfs[i][2]-DDfs[i][1,1]) for i in 1:nCollocationPts) 
        # Boundary conditions:
        + NNlib.mean(abs2.(ϕmodel(∂xpts,params)[:] - ϕx0)) 
        # ϕ initial value (t=0)
        + NNlib.mean(abs2.(ϕmodel(∂t0pts,params)[:] - ϕt0))
        )
end
@info "Loss defined"

# uncomment to load in trained parameters
# loaded_params = jldopen("diffusion_eq_NN_params.jld2")["params"]

optf = OptimizationFunction((p,_)->diffusionLoss(ϕmodel,p), Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, initParams)
# prob = OptimizationProblem(optf, loaded_params)
# prob = OptimizationProblem(optf, sol.u)
@time sol = solve(prob,OptimizationOptimJL.BFGS(),maxiters=200,callback=((_,lossval) -> (println("loss = ",lossval); false)))

# save parameters:
# jldsave("diffusion_eq_NN_params.jld2"; params=sol.u);

@info "Done solving and saving"


using Plots, PlotThemes, LaTeXStrings; theme(:dao)

let xs = 0:0.01:1, ts = 0:0.01:Tmax
#     vals = reshape([ϕmodel([x, t], initParams)[1] for x in xs for t in ts],(length(xs),length(ts)));
    vals = reshape([ϕmodel([x, t], sol.u)[1] for x in xs for t in ts],(length(xs),length(ts)));
#     vals = reshape([ϕmodel([x, t], loaded_params)[1] for x in xs for t in ts],(length(xs),length(ts)));
    plot(ts,xs,vals,linetype=:contourf)
    scatter!(eachrow(collocationPts)...,label=:none)
    scatter!(eachrow(∂t0pts)...,label=:none)
    scatter!(eachrow(∂xpts)...,label=:none)
    plot!(xlabel=L"x",ylabel=L"t")
end


plot()
for t in 0:0.1:Tmax
    pltpts = (collect∘transpose∘hcat)(xmax*range(0,1,100),t*ones(100));
#     plot!(pltpts[1,:],vec(ϕmodel(pltpts, initParams)))
    plot!(pltpts[1,:],vec(ϕmodel(pltpts, sol.u)),label=:none)
#     plot!(pltpts[1,:],vec(ϕmodel(pltpts, loaded_params)),label=:none)
end
scatter!(∂t0pts[1,:],ϕt0,label=:none)
plot!(xlabel=L"x",ylabel=L"u(t,x)")

