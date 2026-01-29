using CSV
using DataFrames
using Random
using Statistics
using Plots

include(joinpath(@__DIR__, "../common/fitness.jl"))
include(joinpath(@__DIR__, "../common/individual.jl"))
include(joinpath(@__DIR__, "../common/mutate.jl"))
include(joinpath(@__DIR__, "../common/crossover.jl"))
include(joinpath(@__DIR__, "../common/plot.jl"))

file_path = joinpath(@__DIR__, "knapPI_12_500_1000_82.csv")
df = CSV.read(file_path, DataFrame)

const WEIGHTS = df.w
const PRICES = df.p

# The max_ration is "price per unit of weight" of the most cost-effective item.
# It makes sure the penalty always is large enough for all items, and that the algorithm
# cannot "cheat" by gaining value by going over the limit. 
ratios = PRICES ./ WEIGHTS
max_ratio = maximum(ratios)

# Adding +1.0 so that taking an item if you are over the limit will
# result in a net loss, even if its the most expensive item per weight unit
const PENALTY_FACTOR = max_ratio + 1.0


function penalty(weight::Int64)
    if weight > KNAPSACK_CAPACITY
        overweight_amount = weight - KNAPSACK_CAPACITY
        penalty = overweight_amount * PENALTY_FACTOR
        
        return penalty
    else
        return 0
    end
end


function fitness_score(arr::BitVector)
    total_price::Int64 = 0
    total_weight::Int64 = 0

    for (i, bit) in enumerate(arr)
        if bit
            total_price += PRICES[i]
            total_weight += WEIGHTS[i]
        end
    end

    final_score = total_price - penalty(total_weight)
    
    return max(0.1, final_score)
end


"""
parent_selection() uses FPS with roulette wheel selection. 
"""
function parent_selection(population::Vector{Individual})
    fitness_sum = calculate_fitness_sum(population)

    for ind in population
        ind.selection_prob = ind.fitness / fitness_sum 
    end

    parent_pool = Individual[]

    for _ in 1:POPULATION_SIZE

        prob_sum = 0.0
        prob_threshold = rand()

        for ind in population
            prob_sum += ind.selection_prob

            if prob_sum > prob_threshold
                push!(parent_pool, ind)
                break
            end
        end
    end

    return parent_pool
end


function generate_next_gen(parent_pool::Vector{Individual})
    new_pop = []   

    for i in 1:2:length(parent_pool)
        parent1 = parent_pool[i]
        parent2 = parent_pool[i+1]

        child_genes = crossover(parent1.genes, parent2.genes)

        mutate!(child_genes[1])
        mutate!(child_genes[2])

        child1 = Individual(child_genes[1])
        child2 = Individual(child_genes[2])

        child1.fitness = fitness_score(child_genes[1])
        child2.fitness = fitness_score(child_genes[2])

        append!(new_pop, (child1, child2))
    end

    return new_pop
end

# Helper functions
get_weight(genes::BitVector) = sum(WEIGHTS[i] for (i, bit) in enumerate(genes) if bit)
valid_solution(weight::Int64) = KNAPSACK_CAPACITY - weight >= 0


"""
main() runs the full genetic algorithm loop.

Cycle per generation:
1) Evaluate fitness for every individual.
2) Keep only valid solutions (weight <= capacity) for reporting best/mean/min/max.
3) Track the best valid solution seen so far (global best).
4) Select parents with roulette-wheel selection (based on fitness).
5) Create the next generation via crossover + mutation.
6) Store stats for plotting and print progress occasionally.

After all generations, prints the best valid solution found and plots fitness curves.
"""
function main()
    mean_scores = []
    minimum_scores = []
    maximum_scores = [] 

    global_best_ind = nothing
    global_best_fitness = 0.0

    population::Vector{Individual} = initialize_population(POPULATION_SIZE, GENES_SIZE)

    # Calculates initial fitness scores
    for ind in population
        ind.fitness = fitness_score(ind.genes)
    end

    for i in 1:MAX_GENERATIONS
        # Filter out invalid genes
        valid_pop = [ind for ind in population if get_weight(ind.genes) <= KNAPSACK_CAPACITY]

        if !isempty(valid_pop)
            current_max = maximum(ind.fitness for ind in valid_pop)
            current_min = minimum(ind.fitness for ind in valid_pop)
            current_mean = mean(ind.fitness for ind in valid_pop)

            if current_max > global_best_fitness
                best_in_gen = valid_pop[argmax([ind.fitness for ind in valid_pop])]
                global_best_fitness = current_max
                global_best_ind = deepcopy(best_in_gen)
            end
        else
            current_max = 0.0
            current_min = 0.0
            current_mean = 0.0
        end

        push!(maximum_scores, current_max)
        push!(minimum_scores, current_min)
        push!(mean_scores, current_mean)

        if i % 50 == 0
            println("Mean fitness for gen $i is: $(round(Int, current_mean))")
        end

        parent_pool = parent_selection(population)
        population = generate_next_gen(parent_pool)
    end

    println("\n=== RESULTS ===")
    if global_best_ind !== nothing
        final_weight = get_weight(global_best_ind.genes)
        println("Best Valid Fitness Found: $(global_best_ind.fitness)")
        println("Weight: $final_weight / $KNAPSACK_CAPACITY \n")
    else
        println("No valid solution found. \n")
    end

    plot_fitness(mean_scores, minimum_scores, maximum_scores)

    # Prevents the plot window from closing immediately
    println("Press Enter to exit...")
    readline()
end


# Simulation Parameters
const KNAPSACK_CAPACITY::Int64 = 280785
const POPULATION_SIZE::Int64 = 500
const GENES_SIZE::Int64 = 500
const CROSSOVER_PROB::Float64 = 0.6
const MUTATION_RATE::Float64 = 0.0001
const MAX_GENERATIONS::Int64 = 2000

# Prevents the main script from running when being imported in another file
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

        
