using CSV
using DataFrames
using Random
using Statistics
using Plots

Random.seed!(123)

file_path = joinpath(@__DIR__, "knapPI_12_500_1000_82.csv")
df = CSV.read(file_path, DataFrame)


# ============================================================================
# GA Hyperparameters
# I guess genes_size isn't really a hyperparameter, but it is defined here for convenience
# ============================================================================
const KNAPSACK_CAPACITY::Int64 = 280785
const POPULATION_SIZE::Int64 = 500
const GENES_SIZE::Int64 = 500
const CROSSOVER_PROB::Float64 = 0.8
const MUTATION_RATE::Float64 = 0.0001
const MAX_GENERATIONS::Int64 = 10000


# ============================================================================
# Population initialization + individual object
# ============================================================================

mutable struct Individual
    genes::BitVector
    fitness::Float64
    selection_prob::Float64
end

Individual(size::Int) = Individual(Random.bitrand(size), 0.0, 0.0)
Individual(genes::BitVector) = Individual(genes, 0.0, 0.0)

function initialize_population(pop_size, genes_size)
    population = [Individual(genes_size) for _ in 1:pop_size]
    return population
end


# ============================================================================
# Fitness-calculation
# ============================================================================

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

"""
Returns the penalty for exceeding the knapsack capacity.
"""
function penalty(weight::Int64)
    if weight > KNAPSACK_CAPACITY
        overweight_amount = weight - KNAPSACK_CAPACITY
        penalty = overweight_amount * PENALTY_FACTOR
        
        return penalty
    else
        return 0
    end
end


"""
Calculates the fitness score for a given BitVector of genes.
The fitness is the total price of the selected items minus any penalty for going over capacity.
"""
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

calculate_fitness_sum(population) = sum(ind.fitness for ind in population)
mean_fitness(pop) = mean(ind.fitness for ind in pop)


# ============================================================================
# Parent pool selection + next-gen generation
# ============================================================================
"""
parent_selection() uses FPS with roulette wheel selection. 
Returns a parent pool of the same size as the original population.
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


"""
generate_next_gen() creates children from the parent pool using crossover and mutation,
then applies elitism to form the new population. 1% of the best individuals from the old population are kept.
"""
function generate_next_gen(parent_pool::Vector{Individual})
    children::Vector{Individual} = []   

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

        append!(children, (child1, child2))
    end

    new_population = elitism(parent_pool, children, Int(0.01 * POPULATION_SIZE))

    return new_population
end




# ============================================================================
# Crossover
# ============================================================================
"""
Crossover function that combines two parent BitVectors to produce offspring
based on a crossover probability.
"""
function crossover(parent1_vec::BitVector, parent2_vec::BitVector)
    if rand() < CROSSOVER_PROB
        cut_point = rand(1:GENES_SIZE)
        child1 = vcat(parent1_vec[1:cut_point], parent2_vec[cut_point+1:end])
        child2 = vcat(parent2_vec[1:cut_point], parent1_vec[cut_point+1:end])

        return [child1, child2]
    else
        return (copy(parent1_vec), copy(parent2_vec))
    end
end


# ============================================================================
# Mutation
# ============================================================================
"""
Mutates the genes of an individual BitVector in place based on the MUTATION_RATE
"""
function mutate!(genes::BitVector)
    for i in eachindex(genes)
        if rand() < MUTATION_RATE
            genes[i] = 1 - genes[i]
        end
    end
end


# ============================================================================
# Survivor selection
# ============================================================================
"""
Combines the old population and children, then selects the top individuals based on fitness.
The population returned is the new generation.
"""
function global_elitism(old_pop::Vector{Individual}, children::Vector{Individual})
    pop_size = length(old_pop)
    
    combined_pool = vcat(old_pop, children)
    sort!(combined_pool, by = ind -> ind.fitness, rev=true)
    
    return combined_pool[1:pop_size]
end


"""
Returns a new population by keeping the elite individuals from the old population
and filling the rest with the best individuals from the children.
"""
function elitism(old_pop::Vector{Individual}, children::Vector{Individual}, elite_count::Int)
    sort!(old_pop, by = ind -> ind.fitness, rev=true)
    elites = old_pop[1:elite_count]
    
    sort!(children, by = ind -> ind.fitness, rev=true)
    non_elites = children[1:(length(children) - elite_count)]
    
    return vcat(elites, non_elites)
end


# ===========================================================================
# Helper functions
# ===========================================================================
"""
Helper functions to get weight and check validity of a solution.
"""
get_weight(genes::BitVector) = sum(WEIGHTS[i] for (i, bit) in enumerate(genes) if bit)
valid_solution(weight::Int64) = KNAPSACK_CAPACITY - weight >= 0


# ============================================================================
# Plotting
# ============================================================================
"""
plot_fitness() is only used in the knapsack problem. It plots the fitness scores
over generations, highlighting the best score and generation if provided.
"""
function plot_fitness(
    mean_scores, 
    minimum_scores, 
    maximum_scores, 
    best_score=nothing, 
    best_generation=nothing; 
    outfile="Project 1/knapsack/fitness_plot.pdf")
    
    plot(mean_scores,
         label="Mean Fitness",
         xlabel="Generation",
         ylabel="Fitness Score",
         title="Genetic Algorithm Performance",
         lw=2,
         size=(800, 600))

    plot!(maximum_scores, label="Max Fitness", color=:green)
    plot!(minimum_scores, label="Min Fitness", color=:red)

    if best_score !== nothing && best_generation !== nothing
        best_score_label = best_score isa Integer ? string(best_score) : string(round(best_score, digits=3))
        vline!([best_generation],
               label="Best $(best_score_label) - gen $(best_generation)",
               linestyle=:dash,
               color=:black,
               lw=2)

        scatter!([best_generation], [best_score],
                 label=false,
                 markershape=:star5,
                 color=:black)
    end

    savefig(outfile)
    return nothing
end



# ============================================================================
# Main GA loop
# ============================================================================
"""
main() runs the full genetic algorithm loop.
It filters out invalid solutions when tracking the best fitness scores, but
uses the whole population when calculating mean and minimum fitness scores.
"""
function main()
    mean_scores = []
    minimum_scores = []
    maximum_scores = [] 

    global_best_ind = nothing
    global_best_fitness = 0
    global_best_generation = nothing

    population::Vector{Individual} = initialize_population(POPULATION_SIZE, GENES_SIZE)

    # Calculates initial fitness scores
    for ind in population
        ind.fitness = fitness_score(ind.genes)
    end

    for i in 1:MAX_GENERATIONS
        # Filter out invalid genes when looking for best fitness
        valid_pop = [ind for ind in population if get_weight(ind.genes) <= KNAPSACK_CAPACITY]
        current_max = maximum(ind.fitness for ind in valid_pop)

        # Calculate other stats using the whole population
        current_min = minimum(ind.fitness for ind in population)
        current_mean = mean(ind.fitness for ind in population)

        if current_max > global_best_fitness
            best_in_gen = valid_pop[argmax([ind.fitness for ind in valid_pop])]
            global_best_fitness = current_max
            global_best_ind = deepcopy(best_in_gen)
            global_best_generation = i
        end

        push!(maximum_scores, current_max)
        push!(minimum_scores, current_min)
        push!(mean_scores, current_mean)

        if i % 100 == 0
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

    plot_fitness(mean_scores, minimum_scores, maximum_scores, global_best_fitness, global_best_generation)

    # Prevents the plot window from closing immediately
    println("Press Enter to exit...")
    readline()
end

# Run the genetic algorithm
main()