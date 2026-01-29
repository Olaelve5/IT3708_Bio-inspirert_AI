include(joinpath(@__DIR__, "individual.jl"))

# Memorize fitness scores to prevent unnecessary calculations
const fitness_memory = Dict{BitVector, Float64}()

"""
This function is only used in the feature selection problem.
Knapsack has its own fitness function defined in knapsack.jl
"""
function fitness_score(genes::BitVector)
    # Only give a penalty if it doesn't take any features
    if sum(genes) == 0
        return 0.0001
    end

    # If we have already seen these genes, don't calculate again
    if haskey(fitness_memory, genes)
        rmse = fitness_memory[genes]
    else
        X_sub = get_columns(X, genes)
        rmse = get_fitness(model, X_sub, y)
        fitness_memory[genes] = rmse
    end

    # Inverts the fitness score (smaller error give higher score)
    # 1e-6 adds a tiny value to the error to prevent division by 0
    final_score = 1 / (rmse + 1e-6)

    return final_score
end

calculate_fitness_sum(population) = sum(ind.fitness for ind in population)
mean_fitness(pop) = mean(ind.fitness for ind in pop)