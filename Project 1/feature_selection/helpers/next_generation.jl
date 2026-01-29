include(joinpath(@__DIR__, "individual.jl"))

"""
parent_selection() uses tournament selection as FPS with roulette wheel selection
did not perform well for this problem.
"""
function parent_selection(population::Vector{Individual})
    group_size = TOURNAMENT_GROUP_SIZE
    parent_pool::Vector{Individual} = []

    for _ in 1:POPULATION_SIZE
        group_indices = rand(1:POPULATION_SIZE, group_size)
        winner_index = argmax(i -> population[i].fitness, group_indices)

        push!(parent_pool, population[winner_index])
    end

    return parent_pool
end


"""
generate_next_gen() creates the next generation from the current population using crossover and mutation.
It also applies crowding if specified.
"""
function generate_next_gen(population::Vector{Individual}, crowding_mode::Symbol=:none)
    new_pop = Vector{Individual}()

    for i in 1:2:length(population)
        parent1 = population[i]
        parent2 = population[i+1]

        child_genes = crossover(parent1.genes, parent2.genes)

        mutate!(child_genes[1])
        mutate!(child_genes[2])

        child1 = Individual(child_genes[1])
        child2 = Individual(child_genes[2])

        child1.fitness = fitness_score(child_genes[1])
        child2.fitness = fitness_score(child_genes[2])

        if crowding_mode == :probabilistic_crowding
            survivor1, survivor2 = probabilistic_crowding([parent1, parent2], [child1, child2])
        elseif crowding_mode == :deterministic_crowding
            survivor1, survivor2 = deterministic_crowding([parent1, parent2], [child1, child2])
        else
            survivor1 = child1
            survivor2 = child2
        end

        append!(new_pop, (survivor1, survivor2))
    end

    return new_pop
end