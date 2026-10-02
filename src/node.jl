"""
    AbtractFrankWolfeNode <: AbstractNode 
    NodeInfo

Holds the necessary information of every node.
This needs to be added by every `AbstractNode` as `std::NodeInfo`

This variant is more flexible than Bonobo.BnBNodeInfo.
"""
abstract type AbstractFrankWolfeNode <: AbstractNode end

"""
    set_node_bound!(objective_sense::Symbol, node::AbstractNode, lb, ub)

Set the bounds of the `node` object to the lower and upper bound given. 
Internally everything is stored as a minimization problem. Therefore the objective_sense `:Min`/`:Max` is needed.
"""
function set_node_bound!(objective_sense::Symbol, node::AbstractNode, lb, ub)
    if isnan(ub)
        ub = Inf
    end
    if objective_sense == :Min
        node.lb = max(lb, node.lb)
        node.ub = ub
    else
        node.lb = max(-lb, node.lb)
        node.ub = -ub
    end
end

"""
    bound!(tree::BnBTree, current_node_id::Int)

Close all nodes which have a lower bound higher or equal to the incumbent.
"""
function bound!(tree::BnBTree, current_node_id::Int)
    for (_, node) in tree.nodes
        if node.id != current_node_id && node.lb >= tree.incumbent
            close_node!(tree, node)
        end
    end
end

"""
    close_node!(tree::BnBTree, node::AbstractNode)

Delete the node from the nodes dictionary and the priority queue.
"""
function close_node!(tree::BnBTree, node::AbstractNode)
    delete!(tree.nodes, node.id)
    return delete!(tree.node_queue, node.id)
end

# ---- branch! dispatch (only this bit changes) ----
function branch!(tree, node)
    orbits_fn = get(tree.root.options, :orbital_orbits, nothing)
    if orbits_fn !== nothing
        return branch_orbital!(tree, node, orbits_fn)
    end

    variable_idx = get_branching_variable(tree, tree.options.branch_strategy, node)
    variable_idx == -1 && return
    nodes_info = get_branching_nodes_info(tree, node, variable_idx)
    for node_info in nodes_info
        add_node!(tree, node, node_info)
    end
end



# ---- filter an active set by a set of (lower, upper) fixings ----
function filter_active_set_by_bounds(active_set, tree, lower, upper)
    atol = tree.options.atol
    AS = typeof(active_set)

    kept = Tuple{Float64,eltype(active_set.atoms)}[]
    for (λ, v) in active_set
        ok = true
        for (i, lo) in lower
            if v[i] < lo - atol; ok = false; break; end
        end
        if ok
            for (i, hi) in upper
                if v[i] > hi + atol; ok = false; break; end
            end
        end
        ok && push!(kept, (λ, v))
    end

    isempty(kept) && return nothing
    total = sum(first, kept)
    return AS([(w / total, v) for (w, v) in kept])
end

"""
    apply_orbital_fixing!(varbounds, n_vertices::Int)

Orbital fixing (Theorem 3, Ostrowski et al., Math. Prog. 2009).

Given a child's `varbounds` and the number of graph vertices, computes the
orbital partition of the group `G(A(F_1, ∅))` — i.e. only edges fixed to 1
are treated as fixed, all other edges (including those fixed to 0) are
considered unassigned. For each orbit that contains a member with
`upper_bound == 0` in the actual child, all remaining members of that orbit
are fixed to zero.

Mutates `varbounds` in place and returns the set `S` of newly-fixed indices.
"""
function apply_orbital_fixing!(varbounds::IntegerBounds, n_vertices::Int)
    n_edges = n_vertices * (n_vertices - 1) ÷ 2

    lb = zeros(Int, n_edges)
    ub = ones(Int,  n_edges)
    for (i, v) in varbounds.lower_bounds
        lb[i] = Int(round(v))
    end
    for (i, v) in varbounds.upper_bounds
        ub[i] = Int(round(v))
    end

    # Orbits of G(A(F_1, ∅)): pretend F_0 is empty for the symmetry computation.
    ones_ub = ones(Int, n_edges)
    orbits  = compute_unassigned_orbits(n_vertices, lb, ones_ub)

    S = Int[]
    for orbit in orbits
        if any(e -> ub[e] == 0, orbit)          # orbit intersects F_0
            for e in orbit
                if ub[e] != 0                    # e is still free in this child
                    push!(S, e)
                end
            end
        end
    end

    for e in S
        if haskey(varbounds.lower_bounds, e)
            delete!(varbounds.lower_bounds, e)
        end
        varbounds.upper_bounds[e] = 0.0
    end

    return S
end

"""
    get_branching_variable(tree::BnBTree, ::MOST_INFEASIBLE, node::AbstractNode)

Return the branching variable which is furthest away from being feasible based on [`get_distance_to_feasible`](@ref)
or `-1` if all integer constraints are respected.
"""
function get_branching_variable(tree::BnBTree, ::MOST_INFEASIBLE, node::AbstractNode)
    values = get_relaxed_values(tree, node)
    best_idx = -1
    max_distance_to_feasible = 0.0
    for i in tree.branching_indices
        value = values[i]
        if !is_approx_feasible(tree, value)
            distance_to_feasible = get_distance_to_feasible(tree, value)
            if distance_to_feasible > max_distance_to_feasible
                best_idx = i
                max_distance_to_feasible = distance_to_feasible
            end
        end
    end
    return best_idx
end

"""
    set_root!(tree::BnBTree, node_info::NamedTuple)

Set the root node information based on the `node_info` which needs to include the same fields as the `Node` struct given 
to the [`initialize`](@ref) method. (Besides the `std` field which is set by Bonobo automatically)

# Example
If your node structure is the following:
```julia
mutable struct MIPNode <: AbstractNode
    std :: BnBNodeInfo
    lbs :: Vector{Float64}
    ubs :: Vector{Float64}
    status :: MOI.TerminationStatusCode
end
```

then you can call the function with this syntax:

```julia
set_root!(tree, (
    lbs = fill(-Inf, length(x)),
    ubs = fill(Inf, length(x)),
    status = MOI.OPTIMIZE_NOT_CALLED
))
```
"""
function set_root!(tree::BnBTree, node_info::NamedTuple)
    return add_node!(tree, nothing, node_info)
end

"""
    add_node!(tree::BnBTree{Node}, parent::Union{AbstractNode, Nothing}, node_info::NamedTuple)

Add a new node to the tree using the `node_info`. For information on that see [`set_root!`](@ref).
"""
function add_node!(
    tree::BnBTree{Node},
    parent::Union{AbstractNode,Nothing},
    node_info::NamedTuple,
) where {Node<:AbstractNode}
    node_id = tree.num_nodes + 1
    node = create_node(Node, node_id, parent, node_info)
    # only add the node if it's better than the current best solution
    if node.lb < tree.incumbent
        tree.nodes[node_id] = node
        tree.node_queue[node_id] = (node.lb, node_id)
        tree.num_nodes += 1
    end
end

"""
    create_node(Node, node_id::Int, parent::Union{AbstractNode, Nothing}, node_info::NamedTuple)

Creates a node of type `Node` with id `node_id` and the named tuple `node_info`. 
For information on that see [`set_root!`](@ref).
"""
function create_node(Node, node_id::Int, parent::Union{AbstractNode,Nothing}, node_info::NamedTuple)
    lb = -Inf
    depth = 1
    if !isnothing(parent)
        lb = parent.lb
        depth = parent.depth + 1
    end
    bnb_node = structfromnt(BnBNodeInfo, (id=node_id, lb=lb, ub=Inf, depth=depth))
    bnb_nt = (std=bnb_node,)
    node_nt = merge(bnb_nt, node_info)
    return structfromnt(Node, node_nt)
end

"""
    get_next_node(tree::BnBTree, ::BestFirstSearch)

Get the next node of the tree which shall be evaluted next by [`evaluate_node!`](@ref).
If you want to implement your own traversing strategy check out [`AbstractTraverseStrategy`](@ref).
"""
function get_next_node(tree::BnBTree, ::BestFirstSearch)
    node_id, _ = first(tree.node_queue)
    return tree.nodes[node_id]
end

function get_next_node(tree::BnBTree, ::DepthFirstSearch)
    node_id = argmax(k -> tree.nodes[k].depth, keys(tree.nodes))
    return tree.nodes[node_id]
end


"""
    FrankWolfeNode <: AbstractFrankWolfeNode

A node in the branch-and-bound tree storing information for a Frank-Wolfe subproblem.

`std` stores the id, lower, upper bound and Depth of the node.
`active_set` store the active set structure.
`local_bounds` instead of storing the complete LMO, it just stores the bounds specific to THIS node.
    All other integer bounds are stored in the root.
'level' stores the level in the tree
'fw_dual_gap_limit' set the tolerance for the dual gap in the FW algorithms
'pre_computed_set' stores specifically the extreme points computed in DICG for warm-start.
'parent_lower_bound_base' contains lower bound value of the parent node.  Needed
    for updating pseudocosts.
'branched_on' contains the index of the parent. Required for updating pseudocosts.
'branched_right' Boolean value specifying if node resulted from a left or right branch. Needed
    for updating pseudocosts.
'distance_to_int' Stores information on the rounding amount at branching. Required
    for correct scaling of pseudocosts.

"""
mutable struct FrankWolfeNode{
    AT<:FrankWolfe.ActiveSet,
    DVS<:FrankWolfe.DeletedVertexStorage,
    IB<:IntegerBounds,
    NI<:NodeInfo,
} <: AbstractFrankWolfeNode
    std::NI
    active_set::AT
    discarded_vertices::DVS
    local_bounds::IB
    fw_dual_gap_limit::Float64
    fw_time::Millisecond
    global_tightenings::Int
    local_tightenings::Int
    local_potential_tightenings::Int
    dual_gap::Float64
    pre_computed_set::Any
    parent_lower_bound_base::Float64
    branched_on::Int
    branched_right::Bool
    distance_to_int::Float64
    active_set_size::Int
    discarded_set_size::Int
end


# For i.e. pseudocost branching we require additional information to be stored in FrankWolfeNode
# this information can be set to a default value if not needed.
FrankWolfeNode(
    std,
    active_set,
    discarded_vertices,
    local_bounds,
    fw_dual_gap_limit,
    fw_time,
    global_tightenings,
    local_tightenings,
    local_potential_tightenings,
    dual_gap,
    pre_computed_set,
) = FrankWolfeNode(
    std,
    active_set,
    discarded_vertices,
    local_bounds,
    fw_dual_gap_limit,
    fw_time,
    global_tightenings,
    local_tightenings,
    local_potential_tightenings,
    dual_gap,
    pre_computed_set,
    Inf,
    -1,
    false,
    0.0,
    0,
    0,
)


"""
Create the information of the new branching nodes 
based on their parent and the index of the branching variable
"""
function get_branching_nodes_info(tree::BnBTree, node::FrankWolfeNode, vidx::Int)
    if !is_valid_split(tree, vidx)
        error("Splitting on the same index as parent! Abort!")
    end

    # get iterate, primal and lower bound
    x = get_relaxed_values(tree, node)
    primal = tree.root.problem.f(x)
    lower_bound_base = primal - node.dual_gap
    @assert isfinite(lower_bound_base)
    left_distance = x[vidx] - floor(x[vidx])
    right_distance = ceil(x[vidx]) - x[vidx]

    user_prune_left, user_prune_right = false, false

    if tree.root.options[:branch_callback] !== nothing
        user_prune_left, user_prune_right = tree.root.options[:branch_callback](tree, node, vidx)
    end

    # In case of strong convexity, check if a child can be pruned
    prune_left, prune_right = if !tree.root.options[:no_pruning]
        prune_children(tree, node, lower_bound_base, x, vidx)
    else
        false, false
    end

    #different ways to split active set
    if !(typeof(tree.root.options[:variant]) <: DecompositionInvariant)

        # Keep the same pre_computed_set
        pre_computed_set_left, pre_computed_set_right = node.pre_computed_set, node.pre_computed_set

        # Split active set
        active_set_left, active_set_right =
            split_vertices_set!(node.active_set, tree, vidx, node.local_bounds)
    else
        if node.pre_computed_set !== nothing
            # Split pre_computed_set
            pre_computed_set_left, pre_computed_set_right =
                split_pre_computed_set!(x, node.pre_computed_set, tree, vidx, node.local_bounds)
        else
            pre_computed_set_left, pre_computed_set_right =
                node.pre_computed_set, node.pre_computed_set
        end
        active_set_left, active_set_right = node.active_set, node.active_set
    end

    discarded_set_left, discarded_set_right =
        split_vertices_set!(node.discarded_vertices, tree, vidx, x, node.local_bounds)

    if !(typeof(tree.root.options[:variant]) <: DecompositionInvariant)
        # Sanity check
        @assert isapprox(sum(active_set_left.weights), 1.0) "sum weights left: $(sum(active_set_left.weights))"
        @assert sum(active_set_left.weights .< 0) == 0
        for v in active_set_left.atoms
            if !(v[vidx] <= floor(x[vidx]) + tree.options.atol)
                error("active_set_left\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
            end
        end
        @assert isapprox(sum(active_set_right.weights), 1.0) "sum weights right: $(sum(active_set_right.weights))"
        @assert sum(active_set_right.weights .< 0) == 0
        for v in active_set_right.atoms
            if !(v[vidx] >= ceil(x[vidx]) - tree.options.atol)
                error("active_set_right\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
            end
        end
        for v in discarded_set_left.storage
            if !(v[vidx] <= floor(x[vidx]) + tree.options.atol)
                error("storage left\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
            end
        end
        for v in discarded_set_right.storage
            if !(v[vidx] >= ceil(x[vidx]) - tree.options.atol)
                error("storage right\n$(v)\n$vidx, $(x[vidx]), $(v[vidx])")
            end
        end
    end

    # add new bounds to the feasible region left and right
    # copy bounds from parent
    varbounds_left = copy(node.local_bounds)
    varbounds_right = copy(node.local_bounds)

    if haskey(varbounds_left.upper_bounds, vidx)
        delete!(varbounds_left.upper_bounds, vidx)
    end
    if haskey(varbounds_right.lower_bounds, vidx)
        delete!(varbounds_right.lower_bounds, vidx)
    end
    push!(varbounds_left.upper_bounds, (vidx => floor(x[vidx])))
    push!(varbounds_right.lower_bounds, (vidx => ceil(x[vidx])))

    # compute new dual gap limit
    fw_dual_gap_limit = tree.root.options[:dual_gap_decay_factor] * node.fw_dual_gap_limit
    fw_dual_gap_limit = max(fw_dual_gap_limit, tree.root.options[:min_node_fw_epsilon])

    if !(typeof(tree.root.options[:variant]) <: DecompositionInvariant)
        # in case of non trivial domain oracle: Only split if the iterate is still domain feasible
        x_left = FrankWolfe.compute_active_set_iterate!(active_set_left)
        x_right = FrankWolfe.compute_active_set_iterate!(active_set_right)

        if !tree.root.options[:domain_oracle](x_left)
            active_set_left =
                build_active_set_by_domain_oracle(active_set_left, tree, varbounds_left, node)
        end
        if !tree.root.options[:domain_oracle](x_right)
            active_set_right =
                build_active_set_by_domain_oracle(active_set_right, tree, varbounds_right, node)
        end
    end

    # update the LMO
    node_info_left = (
        active_set=active_set_left,
        discarded_vertices=discarded_set_left,
        local_bounds=varbounds_left,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
        global_tightenings=0,
        local_tightenings=0,
        local_potential_tightenings=0,
        dual_gap=NaN,
        pre_computed_set=pre_computed_set_left,
        parent_lower_bound_base=lower_bound_base,
        branched_on=vidx,
        branched_right=false,
        distance_to_int=left_distance,
        active_set_size=0,
        discarded_set_size=0,
    )
    node_info_right = (
        active_set=active_set_right,
        discarded_vertices=discarded_set_right,
        local_bounds=varbounds_right,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
        global_tightenings=0,
        local_tightenings=0,
        local_potential_tightenings=0,
        dual_gap=NaN,
        pre_computed_set=pre_computed_set_right,
        parent_lower_bound_base=lower_bound_base,
        branched_on=vidx,
        branched_right=true,
        distance_to_int=right_distance,
        active_set_size=0,
        discarded_set_size=0,
    )

    domain_right = !isempty(active_set_right)
    domain_left = !isempty(active_set_left)

    nodes =
        if !prune_left &&
           !prune_right &&
           domain_right &&
           domain_left &&
           !user_prune_left &&
           !user_prune_right
            [node_info_left, node_info_right]
        elseif prune_left || user_prune_left
            [node_info_right]
        elseif prune_right || user_prune_right
            [node_info_left]
        elseif domain_right # x_right in domain
            [node_info_right]
        elseif domain_left # x_left in domain
            [node_info_left]
        else
            @warn "No childern nodes can be created."
            Vector{typeof(node_info_left)}()
        end
    return nodes
end

"""
Computes the relaxation at that node
"""
function evaluate_node!(tree::BnBTree, node::FrankWolfeNode)
    # check that local bounds and global tightening don't conflict
    for (j, ub) in tree.root.global_tightenings.upper_bounds
        if !haskey(node.local_bounds.lower_bounds, j)
            continue
        end
        lb = node.local_bounds.lower_bounds[j]
        if ub < lb
            @debug "local lb $(lb) conflicting with global tightening $(ub)"
            return NaN, NaN
        end
    end
    for (j, lb) in tree.root.global_tightenings.lower_bounds
        if !haskey(node.local_bounds.upper_bounds, j)
            continue
        end
        ub = node.local_bounds.upper_bounds[j]
        if ub < lb
            @debug "local ub $(ub) conflicting with global tightening $(lb)"
            return NaN, NaN
        end
    end

    # build up node LMO
    build_LMO(
        tree.root.problem.tlmo,
        tree.root.problem.integer_variable_bounds,
        node.local_bounds,
        tree.root.problem.integer_variables,
    )

    # check for feasibility and boundedness
    status = check_feasibility(tree.root.problem.tlmo)
    if status == INFEASIBLE
        @debug "Problem at node $(node.id) infeasible"
        return NaN, NaN
    end
    if status == UNBOUNDED
        error("Feasible region unbounded! Please check your constraints!")
        return NaN, NaN
    end

    decomposition_invariant_starting_point = nothing
    if !(typeof(tree.root.options[:variant]) <: DecompositionInvariant)
        # Check feasibility of the iterate
        active_set = node.active_set
        x = FrankWolfe.compute_active_set_iterate!(node.active_set)
        @assert is_linear_feasible(tree.root.problem.tlmo, x)
        for (_, v) in node.active_set
            @assert is_linear_feasible(tree.root.problem.tlmo, v)
        end
    else
        if node.id == 1 && tree.root.options[:start_solution] !== nothing
            decomposition_invariant_starting_point = tree.root.options[:start_solution]
        elseif tree.root.options[:find_domain_point] !== _trivial_domain_point
            decomposition_invariant_starting_point =
                tree.root.options[:find_domain_point](node.local_bounds)
            if decomposition_invariant_starting_point === nothing
                @debug "Node $(node.id) is infeasible: no domain-feasible starting point found."
                return NaN, NaN
            end
        end
    end

    if tree.root.options[:propagate_bounds] !== nothing
        tree.root.options[:propagate_bounds](tree, node)
    end

    # time tracking FW
    time_ref = Dates.now()
    domain_oracle = tree.root.options[:domain_oracle]

    x, primal, dual_gap, fw_status, atoms_set = solve_frank_wolfe(
        tree.root.options[:variant],
        tree.root.problem.f,
        tree.root.problem.g,
        tree.root.problem.tlmo,
        node.active_set;
        epsilon=node.fw_dual_gap_limit,
        max_iteration=tree.root.options[:max_fw_iter],
        line_search=tree.root.options[:line_search],
        lazy=tree.root.options[:lazy],
        lazy_tolerance=tree.root.options[:lazy_tolerance],
        add_dropped_vertices=tree.root.options[:use_shadow_set],
        use_extra_vertex_storage=tree.root.options[:use_shadow_set],
        extra_vertex_storage=node.discarded_vertices,
        callback=tree.root.options[:callback],
        verbose=tree.root.options[:fw_verbose],
        timeout=tree.root.options[:fw_timeout],
        pre_computed_set=node.pre_computed_set,
        domain_oracle=domain_oracle,
        decomposition_invariant_starting_point=decomposition_invariant_starting_point,
    )

    if tree.root.options[:fw_verbose]
        @show fw_status
    end

    if typeof(atoms_set).name.wrapper == FrankWolfe.ActiveSet
        # update active set of the node
        node.active_set = atoms_set
    else
        node.pre_computed_set = atoms_set
        node.active_set = FrankWolfe.ActiveSet([(1.0, x)])
        # update set of computed atoms and active set
        if isa(x, AbstractVector)
            node.pre_computed_set = atoms_set
            node.active_set = FrankWolfe.ActiveSet([(1.0, x)])
        else
            @debug "x is not a vector, returning NaN, x: $x"
            return NaN, NaN
        end
    end

    node.fw_time = Dates.now() - time_ref
    node.dual_gap = dual_gap

    # tightening bounds at node level
    dual_tightening(tree, node, x, dual_gap)

    # tightening the global bounds
    store_data_global_tightening(tree, node, x, dual_gap)
    global_tightening(tree, node)

    lower_bound = primal - dual_gap
    # tighten the lower bound if the objective is always integral
    lower_bound = tree.root.options[:integral_objective] ? ceil(lower_bound) : lower_bound
    # improvement of the lower bound using strong convexity
    lower_bound = tightening_lowerbound(tree, node, x, lower_bound)

    # Call heuristic 
    run_heuristics(tree, x, tree.root.options[:heuristics])

    # Found an upper bound
    if is_integer_feasible(tree, x)
        node.ub = primal
        @debug "Node $(node.id) has an integer solution."
        return lower_bound, primal
        # Sanity check: If the incumbent is better than the lower bound of the root node
        # and the root node is not integer feasible, something is off!
    elseif node.id == 1 && !tree.root.options[:ignore_lower_bound]
        @debug "Lower bound of root node: $(lower_bound)"
        @debug "Current incumbent: $(tree.incumbent)"
        @assert lower_bound <= tree.incumbent + dual_gap "lower_bound <= tree.incumbent + dual_gap : $(lower_bound) <= $(tree.incumbent + dual_gap)"
    end


    if tree.root.options[:ignore_lower_bound]
        return -Inf, NaN
    end

    node.active_set_size = length(node.active_set)
    node.discarded_set_size = length(node.discarded_vertices)

    return lower_bound, NaN
end

# ---- orbital branching driver ----
function branch_orbital!(tree::BnBTree, node::FrankWolfeNode, orbits_fn)
    orbits = orbits_fn(tree, node)
    isempty(orbits) && return

    variable_idx = get_branching_variable(tree, tree.options.branch_strategy, node)
    variable_idx == -1 && return

    O = nothing
    for orbit in orbits
        if variable_idx in orbit
            O = orbit
            break
        end
    end
    O === nothing && return

    rep = first(O)
    nodes_info = get_orbital_branching_nodes_info(tree, node, rep, O)
    for info in nodes_info
        add_node!(tree, node, info)
    end
    return
end

# ---- two orbital children ----
function get_orbital_branching_nodes_info(
    tree::BnBTree, node::FrankWolfeNode, rep::Int, orbit::Vector{Int},
)
    is_valid_split(tree, rep) ||
        error("Orbital branching: representative $rep was already branched on!")

    x = get_relaxed_values(tree, node)
    primal = tree.root.problem.f(x)
    lower_bound_base = primal - node.dual_gap
    @assert isfinite(lower_bound_base)

    # ---- bound fixings ----
    left_lower  = Dict(rep => 1.0)
    left_upper  = Dict{Int,Float64}()
    right_lower = Dict{Int,Float64}()
    right_upper = Dict(i => 0.0 for i in orbit)

    # ---- pseudocost bookkeeping ----
    left_distance  = x[rep] - floor(x[rep])
    right_distance = minimum(x[i] for i in orbit)

    # ---- user branch callback ----
    user_prune_left, user_prune_right = false, false
    if tree.root.options[:branch_callback] !== nothing
        user_prune_left, user_prune_right =
            tree.root.options[:branch_callback](tree, node, rep)
    end

    # ---- variant-dependent active set handling ----
    is_di = typeof(tree.root.options[:variant]) <: DecompositionInvariant

    if is_di
        # DI: do not split the active set; do not touch pre_computed_set either.
        # The FW loop will re-project using the child's local_bounds.
        active_set_left  = node.active_set
        active_set_right = node.active_set
        pre_left  = node.pre_computed_set
        pre_right = node.pre_computed_set
    else
        active_set_left  = filter_active_set_by_bounds(
            node.active_set, tree, left_lower, left_upper)
        active_set_right = filter_active_set_by_bounds(
            node.active_set, tree, right_lower, right_upper)

        # Empty filter -> fall back to the parent's active set; the child LMO
        # will only accept vertices that satisfy the new local bounds, so the
        # FW loop will simply discard infeasible ones on the first LMO call.
        active_set_left  === nothing && (active_set_left  = node.active_set)
        active_set_right === nothing && (active_set_right = node.active_set)

        pre_left  = node.pre_computed_set
        pre_right = node.pre_computed_set
    end

    # ---- local bounds ----
    varbounds_left  = copy(node.local_bounds)
    varbounds_right = copy(node.local_bounds)
    for (k, v) in left_lower
        haskey(varbounds_left.upper_bounds, k) && delete!(varbounds_left.upper_bounds, k)
        push!(varbounds_left.lower_bounds, (k => v))
    end
    for (k, v) in left_upper
        haskey(varbounds_left.lower_bounds, k) && delete!(varbounds_left.lower_bounds, k)
        push!(varbounds_left.upper_bounds, (k => v))
    end
    for (k, v) in right_lower
        haskey(varbounds_right.upper_bounds, k) && delete!(varbounds_right.upper_bounds, k)
        push!(varbounds_right.lower_bounds, (k => v))
    end
    for (k, v) in right_upper
        haskey(varbounds_right.lower_bounds, k) && delete!(varbounds_right.lower_bounds, k)
        push!(varbounds_right.upper_bounds, (k => v))
    end
    # ---- orbital fixing (Theorem 3) ----
    n_vertices = get(tree.root.options, :orbital_n_vertices, nothing)
    if n_vertices !== nothing
        S_left  = apply_orbital_fixing!(varbounds_left,  n_vertices)
        S_right = apply_orbital_fixing!(varbounds_right, n_vertices)
        isempty(S_left)  || @debug "Orbital fixing (left)  fixed $(length(S_left)) vars:  $S_left"
        isempty(S_right) || @debug "Orbital fixing (right) fixed $(length(S_right)) vars: $S_right"
    end

    fw_dual_gap_limit = max(
        tree.root.options[:dual_gap_decay_factor] * node.fw_dual_gap_limit,
        tree.root.options[:min_node_fw_epsilon],
    )

    # ---- node info ----
    info_left = (
        active_set=active_set_left,
        discarded_vertices=node.discarded_vertices,
        local_bounds=varbounds_left,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
        global_tightenings=0,
        local_tightenings=0,
        local_potential_tightenings=0,
        dual_gap=NaN,
        pre_computed_set=pre_left,
        parent_lower_bound_base=lower_bound_base,
        branched_on=rep,
        branched_right=true,
        distance_to_int=left_distance,
        active_set_size=0,
        discarded_set_size=0,
    )
    info_right = (
        active_set=active_set_right,
        discarded_vertices=node.discarded_vertices,
        local_bounds=varbounds_right,
        fw_dual_gap_limit=fw_dual_gap_limit,
        fw_time=Millisecond(0),
        global_tightenings=0,
        local_tightenings=0,
        local_potential_tightenings=0,
        dual_gap=NaN,
        pre_computed_set=pre_right,
        parent_lower_bound_base=lower_bound_base,
        branched_on=rep,
        branched_right=false,
        distance_to_int=right_distance,
        active_set_size=0,
        discarded_set_size=0,
    )

    domain_left  = !isempty(active_set_left)
    domain_right = !isempty(active_set_right)

    return if domain_left && domain_right &&
              !user_prune_left && !user_prune_right
        [info_left, info_right]
    elseif user_prune_left
        [info_right]
    elseif user_prune_right
        [info_left]
    elseif domain_left
        [info_left]
    elseif domain_right
        [info_right]
    else
        @warn "No orbital children could be created."
        Vector{typeof(info_left)}()
    end
end
