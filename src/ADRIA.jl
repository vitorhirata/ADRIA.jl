module ADRIA

using Requires
using Random, TOML, Dates, CpuId
using StaticArrays, SparseArrays, LinearAlgebra, Statistics, Distributed
using NamedArrays, SparseArrayKit, DifferentialEquations

using MAT
using Combinatorics, Distances
using Setfield, ModelParameters, DataStructures
using DataFrames, Graphs, CSV
import ArchGDAL as AG
import GeoDataFrames

using PkgVersion
using ProgressMeter

using SnoopPrecompile, RelocatableFolders


include("utils/text_display.jl")  # need better name for this file
include("utils/setup.jl")

include("ecosystem/corals/growth.jl")
include("ecosystem/corals/CoralGrowth.jl")
include("ecosystem/Ecosystem.jl")

# Generate base coral struct from default spec.
# Have to call this before including specification methods
create_coral_struct()

include("ecosystem/corals/spec.jl")
include("ecosystem/const_params.jl")

include("io/inputs.jl")
include("Domain.jl")

include("sites/connectivity.jl")
include("sites/dMCDA.jl")

include("interventions/seeding.jl")

include("io/ResultSet.jl")
include("io/result_io.jl")
include("io/result_post_processing.jl")
include("io/sampling.jl")
include("metrics/metrics.jl")
include("metrics/performance.jl")

include("scenario.jl")
include("optimization.jl")
include("analysis/sensitivity.jl")
include("analysis/analysis.jl")


function __init__()
    @require GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a" begin
        @require GeoMakie = "db073c08-6b98-4ee5-b6a4-5efafb3259c6" begin
            @require DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" include("../Aviz/src/Aviz.jl")
        end
    end
end


export fecundity_scope!, bleaching_mortality!
export growthODE
export run_scenario, coral_spec
export create_coral_struct, Intervention, Criteria, Corals, SimConstants
export site_area, site_k_area
export Domain, metrics, select, timesteps, env_stats

# metric helper methods
export dims, ndims

# List out compatible domain datapackages
const COMPAT_DPKG = ["0.3.1"]


@precompile_all_calls begin
    ex_dir = @path joinpath(@__DIR__, "../examples")

    f() = begin
        @showprogress 1 for _ in 1:10
        end
    end
    b = redirect_stdout(f, devnull)

    dom = ADRIA.load_domain(joinpath(ex_dir, "Example_domain"), "45")
    p_df = ADRIA.param_table(dom)
    # p_df = repeat(p_df, 5)
    # p_df[:, :dhw_scenario] .= 50
    # p_df[:, :guided] .= [0, 0, 1, 2, 3]
    # p_df[:, :seed_TA] .= [0, 5e5, 5e5, 5e5, 5e5]
    # p_df[:, :seed_CA] .= [0, 5e5, 5e5, 5e5, 5e5]
    rs1 = ADRIA.run_scenario(p_df[1, :], dom)

    # ENV["ADRIA_THRESHOLD"] = 1e-6
    # run_scenario(p_df[1, :], dom)
    # run_scenario(p_df[end, :], dom)
    # delete!(ENV, "ADRIA_THRESHOLD")

    # precompile(load_results, (String,))
    # precompile(EnvLayer, (String, String, String, String, String, String, String))
end

end
