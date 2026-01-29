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