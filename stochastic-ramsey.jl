using Boscia
using FrankWolfe
using LinearAlgebra, SparseArrays
using Random
include(joinpath(@__DIR__, "src", "objectives.jl"))
using .RamseyObjectives

Random.seed!(42)

function edge_index(n, i, j)
    i == j && error("No diagonal entries")
    if i > j
        i, j = j, i
    end
    offset = (i - 1) * n - ((i - 1) * i) ÷ 2
    return offset + (j - i)
end

function _for_each_combination(f, n::Integer, k::Integer)
    (k < 0 || k > n) && return
    if k == 0
        f(Int[]); return
    end
    comb = collect(1:k)
    while true
        f(comb)
        i = k
        while i >= 1 && comb[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        comb[i] += 1
        for j in (i+1):k
            comb[j] = comb[j-1] + 1
        end
    end
end

struct PenaltyRow
    row::SparseVector{Int, Int}
    is_s_clique::Bool
end

struct PenaltyMatrix
    data::SparseMatrixCSC{Int, Int}
    is_s::BitVector
end

Base.size(pm::PenaltyMatrix) = size(pm.data)
Base.size(pm::PenaltyMatrix, d::Integer) = size(pm.data, d)
Base.length(pm::PenaltyMatrix) = length(pm.is_s)
Base.keys(pm::PenaltyMatrix)   = Base.OneTo(length(pm.is_s))
Base.getindex(pm::PenaltyMatrix, i::Integer) = PenaltyRow(pm.data[i, :], pm.is_s[i])
Base.getindex(pm::PenaltyMatrix, i::Integer, j::Integer) = pm.data[i, j]

#constructor
function PenaltyMatrix(n::Integer, r::Integer, s::Integer)
    d = (n * (n - 1)) ÷ 2
    total_rows = binomial(n, r) + binomial(n, s)
    is_s = BitVector(undef, total_rows)
    I_ = Int[]; J_ = Int[]; V_ = Int[]
    row_idx = Ref(1)
    #r-clique rows
    _for_each_combination(n, r) do clique
        for a in 1:length(clique), b in (a+1):length(clique)
            push!(I_, row_idx[])
            push!(J_, edge_index(n, clique[a], clique[b]))
            push!(V_, 1)
        end
        is_s[row_idx[]] = false
        row_idx[] += 1
    end

    #s-clique rows
    _for_each_combination(n, s) do clique
        for a in 1:length(clique), b in (a+1):length(clique)
            push!(I_, row_idx[])
            push!(J_, edge_index(n, clique[a], clique[b]))
            push!(V_, 1)
        end
        is_s[row_idx[]] = true
        row_idx[] += 1
    end

    data = sparse(I_, J_, V_, total_rows, d)
    return PenaltyMatrix(data, is_s)
end

function main(args=ARGS)
    n = parse(Int, args[1])
    r = parse(Int, args[2])
    s = parse(Int, args[3])
    t = length(args) >= 4 ? parse(Float64, args[4]) : Inf
    cache = build_cache(n, r, s)
    cost(x) = hinge_value(x, cache)
    grad!(storage, x) = hinge_gradient!(storage, x, cache)
    L = curvature_bound(n, r, s)
    N = n * (n - 1) ÷ 2
    
    data = PenaltyMatrix(n, r, s)
    
	function loss(x, sample)
	    idx = sample.row.nzind
	    s = 0.0
	    @inbounds for t in eachindex(idx)
	        s += x[idx[t]]
	    end
	    if sample.is_s_clique
	        return max(0.0, 1.0 - s)^2
	    else
	        return max(0.0, s - (r - 1))^2   # r captured from outer scope
	    end
	end

	function sample_grad!(storage, x, sample)
	    idx = sample.row.nzind
	    s = 0.0
	    @inbounds for t in eachindex(idx)
	        s += x[idx[t]]
	    end
	    if sample.is_s_clique
	        if s < 1.0
	            c = -2.0 * (1.0 - s)
	            @inbounds for t in eachindex(idx)
	                storage[idx[t]] += c
	            end
	        end
	    else
	        if s > r - 1
	            c = 2.0 * (s - (r - 1))
	            @inbounds for t in eachindex(idx)
	                storage[idx[t]] += c
	            end
	        end
	    end
	end
	
	# Construct the stochastic objective
	objective = FrankWolfe.StochasticObjective(
    	loss,
    	sample_grad!,
    	data,
    	zeros(N),
	)
	
	#determining batch size
	nr = binomial(n,r)
	ns = binomial(n,s)
	maxsize = max(nr,ns)
	batch_size = cld(maxsize,400)

    base_lmo = FrankWolfe.BoxLMO(
        zeros(Float64, N),
        ones(Float64, N),
    )

    lmo = Boscia.ManagedLMO(
        base_lmo,
        zeros(Float64, N),
        ones(Float64, N),
        collect(1:N),
        N,
    )

    function build_branch_callback()
        return function (tree, node, vidx::Int)
            x = Boscia.get_relaxed_values(tree, node)         # ← Boscia, not Bonobo
            primal = tree.root.problem.f(x)
            lower_bound = primal - node.dual_gap
            if lower_bound > 0.0 + eps()
                println("No need to branch here. Node lower bound already positive.")
            end
            valid_lower = lower_bound > 0.0 + eps()
            return valid_lower, valid_lower
        end
    end

    function build_tree_callback()
        return function (tree, node;
                         worse_than_incumbent=false,
                         node_infeasible=false,
                         lb_update=false)
            if isapprox(tree.incumbent, 0.0, atol=eps())
                tree.root.problem.solving_stage = Boscia.USER_STOP
                println("Optimal solution found.")
            end
            if Boscia.tree_lb(tree) > 0.0 + eps()             # ← no type annotation
                tree.root.problem.solving_stage = Boscia.USER_STOP
                println("Tree lower bound already positive. No solution possible.")
            end
        end
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = true
    settings.branch_and_bound[:print_iter] = 200
    settings.branch_and_bound[:bnb_callback] = build_tree_callback()
    settings.branch_and_bound[:branch_callback] = build_branch_callback()
    settings.frank_wolfe[:variant] = Boscia.StochasticFrankWolfe(N,objective,batch_size)
    settings.frank_wolfe[:line_search] = FrankWolfe.Agnostic()
    settings.frank_wolfe[:lazy] = false
    settings.frank_wolfe[:max_fw_iter] = 10000
    settings.branch_and_bound[:time_limit] = t


    x, _, result = Boscia.solve(cost, grad!, lmo, settings=settings)

    @show(x)
    @show(cost(x))
    return x, result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
