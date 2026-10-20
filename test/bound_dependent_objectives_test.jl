using Test
using Boscia
using FrankWolfe
using Random
using SCIP
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

@testset "bound dependent objectives" begin
    n = 6
    const diffw = 0.5 * ones(n)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne())
    end
    lmo = FrankWolfe.MathOptLMO(o)

    function f(x)
        return sum(0.5*(x.-diffw).^2)
    end
    function grad!(storage, x)
        @. storage = x-diffw
    end

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = false
    x, _, result1 = Boscia.solve(f, grad!, lmo, settings=settings)

    function my_objective_gradient_buider(integer_variables, lower_bounds, upper_bounds)
        unfixed_variables = Int[]
        num_fixed = 0
        for i in 1:size(integer_variables)
            #variable is fixed to the value 1 or 0
            if lower_bounds[i] = 1 || upper_bounds[i] = 0
                num_fixed += 1
            else
                push!(unfixed_variables,integer_variables[i])
            end
        end
        function my_objective(x)
            unfixed_sum = 0
            for i in unfixed_variables
                unfixed_sum += (x[i] - 0.5)^2
            end
            return 0.5 * ((num_fixed * 0.25) + unfixed_sum)
        end

        function my_gradient!(storage,x)
            #do the node dependent gradient
        end

        return my_objective, my_gradient!
    end

    settings.frank_wolfe[:objective_gradient_builder] = my_objective_gradient_buider
    x, _, result2 = Boscia.solve(f, grad!, lmo, settings=settings)
    @test result1 == result2
end