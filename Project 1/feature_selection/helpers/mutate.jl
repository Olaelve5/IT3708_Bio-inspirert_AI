
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