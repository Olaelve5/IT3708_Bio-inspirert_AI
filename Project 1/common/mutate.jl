function mutate!(genes::BitVector)
    for i in eachindex(genes)
        if rand() < MUTATION_RATE
            genes[i] = 1 - genes[i]
        end
    end
end