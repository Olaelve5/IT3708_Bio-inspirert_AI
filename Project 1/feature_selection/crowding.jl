include(joinpath(@__DIR__, "individual.jl"))


hamming_distance(genes1::BitVector, genes2::BitVector) = count(xor.(genes1, genes2))

function deterministic_crowding(parents::Vector{Individual}, children::Vector{Individual})
    # There are only two ways to pair parents and children
    dist_A = hamming_distance(parents[1].genes, children[1].genes) + 
             hamming_distance(parents[2].genes, children[2].genes)
             
    dist_B = hamming_distance(parents[1].genes, children[2].genes) + 
             hamming_distance(parents[2].genes, children[1].genes)

    if dist_A < dist_B
        s1 = children[1].fitness > parents[1].fitness ? children[1] : parents[1]
        s2 = children[2].fitness > parents[2].fitness ? children[2] : parents[2]
    else
        s1 = children[2].fitness > parents[1].fitness ? children[2] : parents[1]
        s2 = children[1].fitness > parents[2].fitness ? children[1] : parents[2]
    end

    return s1, s2
end



function probabilistic_crowding(parents::Vector{Individual}, children::Vector{Individual})
    dist_A = hamming_distance(parents[1].genes, children[1].genes) + 
             hamming_distance(parents[2].genes, children[2].genes)
             
    dist_B = hamming_distance(parents[1].genes, children[2].genes) + 
             hamming_distance(parents[2].genes, children[1].genes)

    if dist_A < dist_B
        s1 = probabilistic_winner(parents[1], children[1])
        s2 = probabilistic_winner(parents[2], children[2])
    else
        s1 = probabilistic_winner(parents[1], children[2])
        s2 = probabilistic_winner(parents[2], children[1])
    end

    return s1, s2
end


function probabilistic_winner(parent::Individual, child::Individual; scaling_factor=50.0)
    # Apply power scaling to make differences larger
    f_parent_scaled = parent.fitness ^ scaling_factor
    f_child_scaled = child.fitness ^ scaling_factor

    total_fitness = f_parent_scaled + f_child_scaled
    
    if total_fitness == 0.0
        return rand() < 0.5 ? child : parent
    end

    p_child_wins = f_child_scaled / total_fitness
    
    if rand() < p_child_wins
        return child
    else
        return parent
    end
end

