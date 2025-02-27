using
    GeoMakie,
    GeoMakie.Colors

using
    DataFrames,
    DimensionalData,
    YAXArrays

using ADRIA: axes_names, ResultSet, metrics.metric_label, analysis.col_normalize, model_spec
using .AvizExt

const OPT_TYPE = Dict{Symbol,<:Any}
const DEFAULT_OPT_TYPE = Dict{Symbol,Any}

const COLORMAP_TYPE_DOCSTRING = replace("""
Union{
Symbol,RGB{T},RGBA{T},Vector{Symbol},Vector{RGBA{T}},Vector{RGB{T}}
}
""", "\n" => "")
const COLORMAP_TYPE = eval(Meta.parse("f(T)=$COLORMAP_TYPE_DOCSTRING"))

"""
    _time_labels(labels)

Extract time step labels, ensuring last entry is always included.
"""
function _time_labels(labels; label_step=5)::Tuple{Vector{Int64},Vector{String}}
    labels_length = length(labels)
    labels_strings = string.(labels)

    tick_position = collect(1:label_step:labels_length)
    tick_label = collect(labels_strings[1:label_step:end])

    # Prevent missing last label
    if (labels_length - 1) % label_step != 0
        return vcat(tick_position, labels_length), vcat(tick_label, labels_strings[end])
    end

    return tick_position, tick_label
end

"""
    timesteps(outcomes::YAXArray)::Array{Int64}

Extract time step labels from outcome arrays.

# Arguments
- `outcomes` : Results to extract metadata from

# Returns
Array of time steps (years)
"""
function timesteps(outcomes::YAXArray)::Array{Int64}
    axis_labels = axes_names(outcomes)

    if :timesteps in axis_labels
        return Array(outcomes.timesteps)
    end

    return Int64[]
end

"""
    _calc_gridsize(n_factors::Int64; max_cols::Int64=4)::Tuple{Int64,Int64}

Calculates a "nice" number of rows and columns from a given number of factors to display.
The number of rows for subplots are calculated based on the number of desired columns.

Note: `n_factors` == 1 is displayed as a single figure.
      `n_factors` <= 4 are always displayed with 2 columns.

# Arguments
- `n_factors` : Number of factors to organize in a grid.

# Returns
Number of rows and columns
"""
function _calc_gridsize(n_factors::Int64; max_cols::Int64=4)::Tuple{Int64,Int64}
    if n_factors <= 4
        if n_factors == 1
            return 1, 1
        end

        n_cols::Int64 = 2
    else
        n_cols = max_cols
    end

    n_rows::Int64 = ceil(Int64, n_factors / n_cols)

    return n_rows, n_cols
end

include("../outcome_metadata.jl")
include("scenarios.jl")
include("sensitivity.jl")
include("clustering.jl")
include("rule_extraction.jl")
include("location_selection.jl")
include("spatial.jl")
include("taxa_dynamics.jl")
include("environment/dhw.jl")
include("environment/cyclones.jl")
include("data_envelopment.jl")
