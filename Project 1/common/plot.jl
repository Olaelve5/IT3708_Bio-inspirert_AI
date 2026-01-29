using Plots

"""
plot_fitness() is only used in the knapsack problem.
"""
function plot_fitness(mean_scores, minimum_scores, maximum_scores; outfile="Project 1/knapsack/fitness_plot.pdf")
    plot(mean_scores,
         label="Mean Fitness",
         xlabel="Generation",
         ylabel="Fitness Score",
         title="Genetic Algorithm Performance",
         lw=2,
         size=(800, 600))

    plot!(maximum_scores, label="Max Fitness", color=:green)
    plot!(minimum_scores, label="Min Fitness", color=:red)

    savefig(outfile)
    return nothing
end

"""
plot_rmse() is only used in the feature selection problem.
"""
function plot_rmse(mean_scores, minimum_scores, maximum_scores; outfile="Project 1/feature_selection/plots/fitness_plot.pdf")
    plot(mean_scores,
         label="Mean RMSE",
         xlabel="Generation",
         ylabel="RMSE",
         title="Genetic Algorithm Performance",
         lw=2,
         size=(800, 600))

    plot!(maximum_scores, label="Min RMSE", color=:green)
    plot!(minimum_scores, label="Max RMSE", color=:red)

    savefig(outfile)
end

function plot_entropy(entropy_histories::AbstractDict; outfile="Project 1/feature_selection/plots/entropy_plot.pdf", title="Population Entropy Over Generations")
    plt = plot(
        xlabel="Generation",
        ylabel="Entropy",
        title=title,
        lw=2,
        size=(800, 600),
    )

    for (label, history) in pairs(entropy_histories)
        plot!(plt, history, label=string(label))
    end

    savefig(plt, outfile)
end
