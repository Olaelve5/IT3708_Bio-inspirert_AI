
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