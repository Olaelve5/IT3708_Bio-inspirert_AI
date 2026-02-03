# This file includes all helper files for feature selection GA
# - its purpose is to simplify the inclusion of multiple files in main.jl

include(joinpath(@__DIR__, "LinReg.jl"))
include(joinpath(@__DIR__, "entropy.jl"))
include(joinpath(@__DIR__, "crowding.jl"))
include(joinpath(@__DIR__, "plot.jl"))
include(joinpath(@__DIR__, "individual.jl"))
include(joinpath(@__DIR__, "fitness.jl"))
include(joinpath(@__DIR__, "elitism.jl"))
include(joinpath(@__DIR__, "mutate.jl"))
include(joinpath(@__DIR__, "crossover.jl"))
include(joinpath(@__DIR__, "next_generation.jl"))