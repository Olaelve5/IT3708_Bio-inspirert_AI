using CSV
using DataFrames
using StableRNGs
using MLJ

include(joinpath(@__DIR__, "LinReg.jl"))
include(joinpath(@__DIR__, "entropy.jl"))
include(joinpath(@__DIR__, "crowding.jl"))
include(joinpath(@__DIR__, "../common/plot.jl"))
include(joinpath(@__DIR__, "../common/individual.jl"))
include(joinpath(@__DIR__, "../common/fitness.jl"))
include(joinpath(@__DIR__, "../common/survivor.jl"))
include(joinpath(@__DIR__, "../common/mutate.jl"))
include(joinpath(@__DIR__, "../common/crossover.jl"))


function make_regression()
    file_path = joinpath(@__DIR__, "resources/dataset.txt")
    data = CSV.read(file_path, DataFrame, header=0)
    y, X = unpack(data, ==(:Column102))

    return X, y
end

X, y = make_regression()

LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0
LinearRegressor(fit_intercept = true, solver = nothing)
model = LinearRegressor()

myRNG = StableRNG(123)
baseline_rmse = get_fitness(model, X, y; rng=myRNG)

println("Baseline RMSE (All Features): $(round(baseline_rmse, digits=4))")

"""
parent_selection() uses tournament selection as FPS with roulette wheel selection
did not perform well.
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

function run_experiment(mode::Symbol)
    mean_scores = Float64[]
    max_scores = Float64[]
    min_scores = Float64[]
    entropy_history = Float64[]
    
    global_best_ind = nothing
    global_best_fitness = 0.0

    population::Vector{Individual} = initialize_population(POPULATION_SIZE, GENES_SIZE)

    for i in 1:MAX_GENERATIONS
        current_max = maximum(ind.fitness for ind in population)
        current_min = minimum(ind.fitness for ind in population)
        current_mean = mean(ind.fitness for ind in population)

        if current_max > global_best_fitness
            best_in_gen = population[argmax([ind.fitness for ind in population])]
            global_best_fitness = current_max
            global_best_ind = deepcopy(best_in_gen)
        end

        # Convert back to RMSE
        push!(max_scores, (1.0 / current_max))
        push!(min_scores, (1.0 / current_min))
        push!(mean_scores, (1.0 / current_mean))
        push!(entropy_history, calculate_population_entropy(population))

        if i % 5 == 0
            current_rmse = 1.0 / current_mean
            println("Gen $i: Mean RMSE ≈ $(round(current_rmse, digits=4))")
        end

        if mode == :SGA        
            parent_pool = parent_selection(population)
            population = generate_next_gen(parent_pool)
        elseif mode == :deterministic_crowding
            population = generate_next_gen(shuffle(population), :deterministic_crowding)
        elseif mode == :probabilistic_crowding
            population = generate_next_gen(shuffle(population), :probabilistic_crowding)
        elseif mode == :elitism
            parent_pool = parent_selection(population)
            offspring = generate_next_gen(parent_pool)
            population = elitism(population, offspring, 5)
        elseif mode == :global_elitism
            parent_pool = parent_selection(population)
            offspring = generate_next_gen(parent_pool)
            population = global_elitism(population, offspring)
        else
            error("Unknown mode: $mode")
        end
    end

    println("\n=== FEATURE SELECTION RESULTS ===")
    if global_best_ind !== nothing
        best_rmse = 1.0 / global_best_ind.fitness
        
        num_features = sum(global_best_ind.genes)
        
        println("Best RMSE Found:      $best_rmse")
        println("Features Selected:    $num_features / 101")
        println("Fitness Score:        $(global_best_ind.fitness)")
    else
        println("Simulation failed to produce a valid individual.")
    end

    return mean_scores, min_scores, max_scores, entropy_history
end

function main()
    mean_scores_SGA, min_scores_SGA, max_scores_SGA, entropy_history_SGA = run_experiment(:SGA)
    mean_scores_crowding, min_scores_crowding, max_scores_crowding, entropy_history_crowding = run_experiment(:deterministic_crowding)
    mean_scores_elitism, min_scores_elitism, max_scores_elitism, entropy_history_elitism = run_experiment(:elitism)


    plot_rmse(mean_scores_SGA, min_scores_SGA, max_scores_SGA; outfile="Project 1/feature_selection/plots/fitness_plot_SGA.pdf")
    plot_rmse(mean_scores_crowding, min_scores_crowding, max_scores_crowding; outfile="Project 1/feature_selection/plots/fitness_plot_crowding.pdf")
    plot_rmse(mean_scores_elitism, min_scores_elitism, max_scores_elitism; outfile="Project 1/feature_selection/plots/fitness_plot_elitism.pdf")

    plot_entropy(
        Dict(
            "Population Entropy Crowding" => entropy_history_crowding,
            "Population Entropy SGA" => entropy_history_SGA,
            "Population Entropy Elitism" => entropy_history_elitism,
        );
        outfile="Project 1/feature_selection/plots/entropy_plot_crowding_vs_elitism_vs_SGA.pdf",
    )
end

# Simulation Parameters
const POPULATION_SIZE::Int64 = 100
const GENES_SIZE::Int64 = 101
const CROSSOVER_PROB::Float64 = 0.8
const MUTATION_RATE::Float64 = 0.01
const MAX_GENERATIONS::Int64 = 100
const TOURNAMENT_GROUP_SIZE = 3


# Prevents the main script from running when being imported in another file
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end