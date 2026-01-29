using Random


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
