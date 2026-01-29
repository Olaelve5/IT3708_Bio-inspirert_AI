include(joinpath(@__DIR__, "individual.jl"))

"""
Calculates the population entropy based on the distribution of genes.
It uses the entropy formula defined in the assignment.
"""
function calculate_population_entropy(population::Vector{Individual})
    pop_size = length(population)
    gene_length = length(population[1].genes)
    total_entropy = 0.0

    for j in 1:gene_length
        ones_count = sum(ind.genes[j] for ind in population)
        
        p1 = ones_count / pop_size  
        p0 = 1.0 - p1               
        
        if p1 > 0 && p1 < 1.0
            bit_entropy = -(p1 * log2(p1) + p0 * log2(p0))
            total_entropy += bit_entropy
        end
    end

    return total_entropy / gene_length
end