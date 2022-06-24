import Dates: now

using PkgVersion, Zarr, NamedDims

import Setfield: @set!
import DataFrames: DataFrame
import ADRIA: Domain, EnvLayer


const RESULTS = "results"
const LOG_GRP = "logs"
const INPUTS = "inputs"


struct ResultSet{S, T, F}
    name::S
    rcp::Int
    invoke_time::S
    ADRIA_VERSION::S
    site_ids::T
    site_area::F
    site_max_coral_cover::F
    env_layer_md::EnvLayer

    inputs::DataFrame
    sim_constants::Dict

    raw::AbstractArray
    ranks::AbstractArray
    seed_log::AbstractArray
    fog_log::AbstractArray
    shade_log::AbstractArray
end


"""
    store_name(r::ResultSet)::String

Get name of result set.
"""
function store_name(r::ResultSet)::String
    return "$(r.name)__$(r.rcp)__$(r.invoke_time)"
end


"""
    store_location(r::ResultSet)::String

Get location of result set.
"""
function store_location(rs::ResultSet)::String
    store = ""
    try
        store = joinpath(ENV["ADRIA_OUTPUT_DIR"], store_name(rs))
    catch err
        if isa(err, KeyError)
            @warn "Output directory not yet set. Displaying result directory instead."
            store = store_name(rs)
        else
            rethrow(err)
        end
    end

    return store
end


"""
    setup_result_store!(domain::Domain, param_df::DataFrame)

Sets up an on-disk result store.

## Structure of Store

```
├───logs
│   ├───fog
│   ├───rankings
│   ├───seed
│   └───shade
└───results
└───inputs
```

`inputs` store includes domain specification metadata including what connectivity/DHW/wave data was used.

# Notes
- `domain` is replaced with an identical copy with an updated scenario invoke time.
- -9999.0 is used as an arbitrary fill value.

# Inputs
- `domain` : ADRIA scenario domain
- `param_df` : ADRIA scenario specification

# Returns
domain, (raw, site_ranks, seed_log, fog_log, shade_log)

"""
function setup_result_store!(domain::Domain, param_df::DataFrame, reps::Int)::Tuple
    @set! domain.scenario_invoke_time = replace(string(now()), "T"=>"_", ":"=>"_", "."=>"_")
    log_location::String = joinpath(ENV["ADRIA_OUTPUT_DIR"], "$(domain.name)__$(domain.rcp)__$(domain.scenario_invoke_time)")

    z_store = DirectoryStore(log_location)

    # Store copy of inputs
    input_loc::String = joinpath(z_store.folder, INPUTS)
    input_dims::Tuple{Int64,Int64} = size(param_df)
    attrs::Dict = Dict(
        :name => domain.name,
        :rcp => domain.rcp,
        :columns => names(param_df),
        :invoke_time => domain.scenario_invoke_time,
        :ADRIA_VERSION => "v" * string(PkgVersion.Version(@__MODULE__)),
        
        :site_data_file => domain.env_layer_md.site_data_fn,
        :site_id_col => domain.env_layer_md.site_id_col,
        :unique_site_id_col => domain.env_layer_md.unique_site_id_col,
        :init_coral_cover_file => domain.env_layer_md.init_coral_cov_fn,
        :connectivity_file => domain.env_layer_md.connectivity_fn,
        :DHW_file => domain.env_layer_md.DHW_fn,
        :wave_file => domain.env_layer_md.wave_fn,
        :sim_constants => Dict(fn=>getfield(domain.sim_constants, fn) for fn ∈ fieldnames(typeof(domain.sim_constants))),

        :site_ids => unique_sites(domain),
        :site_area => domain.site_data.area,
        :site_max_coral_cover => domain.site_data.k
    )

    inputs = zcreate(Float64, input_dims...; fill_value=-9999.0, fill_as_missing=false, path=input_loc, chunks=input_dims, attrs=attrs)

    # Store post-processed table of input parameters.
    integer_params = findall(domain.model[:ptype] .== "integer")
    map_to_discrete!(param_df[:, integer_params], getindex.(domain.model[:bounds], 2)[integer_params])
    inputs[:, :] = Matrix(param_df)

    # Raw result store
    result_loc::String = joinpath(z_store.folder, RESULTS)
    tf, n_sites = domain.sim_constants.tf::Int64, domain.coral_growth.n_sites::Int64
    result_dims::Tuple{Int64,Int64,Int64,Int64,Int64} = (tf, domain.coral_growth.n_species, n_sites, reps, nrow(param_df))
    
    attrs = Dict(
        :structure => ("timesteps", "species", "sites", "reps", "scenarios"),
        :unique_site_ids => unique_sites(domain)
    )
    compressor = Zarr.BloscCompressor(cname="zstd", clevel=2, shuffle=true)
    raw = zcreate(Float32, result_dims...; fill_value=nothing, fill_as_missing=false, path=result_loc, chunks=(result_dims[1:end-1]..., 1), attrs=attrs, compressor=compressor)

    # Set up logs for site ranks, seed/fog log
    zgroup(z_store, LOG_GRP)
    log_fn::String = joinpath(z_store.folder, LOG_GRP)

    # Store ranked sites
    n_scens::Int64 = nrow(param_df)
    rank_dims::Tuple{Int64,Int64,Int64,Int64} = (tf, n_sites, 2, n_scens)  # sites, site id and rank, reps, no. scenarios
    fog_dims::Tuple{Int64,Int64,Int64,Int64} = (tf, n_sites, reps, n_scens)  # timeframe, sites, reps, no. scenarios

    # tf, no. intervention sites, site id and rank, no. scenarios
    seed_dims::Tuple{Int64,Int64,Int64,Int64,Int64} = (tf, 2, n_sites, reps, n_scens)

    attrs = Dict(
        :structure=> ("timesteps", "sites", "reps", "scenarios"),
        :unique_site_ids=>unique_sites(domain),
    )
    ranks = zcreate(Float32, rank_dims...; name="rankings", fill_value=nothing, fill_as_missing=false, path=log_fn, chunks=(rank_dims[1:3]..., 1), attrs=attrs)

    attrs = Dict(
        :structure=> ("timesteps", "coral_id", "sites", "reps", "scenarios"),
        :unique_site_ids=>unique_sites(domain),
    )
    seed_log = zcreate(Float32, seed_dims...; name="seed", fill_value=nothing, fill_as_missing=false, path=log_fn, chunks=(seed_dims[1:4]..., 1), attrs=attrs)

    attrs = Dict(
        :structure=> ("timesteps", "sites", "reps", "scenarios"),
        :unique_site_ids=>unique_sites(domain),
    )
    fog_log = zcreate(Float32, fog_dims...; name="fog", fill_value=nothing, fill_as_missing=false, path=log_fn, chunks=(fog_dims[1:3]..., 1), attrs=attrs)
    shade_log = zcreate(Float32, fog_dims...; name="shade", fill_value=nothing, fill_as_missing=false, path=log_fn, chunks=(fog_dims[1:3]..., 1), attrs=attrs)

    return domain, (raw=raw, site_ranks=ranks, seed_log=seed_log, fog_log=fog_log, shade_log=shade_log)
end


"""
    load_results(result_loc::String)::ResultSet
    load_results(domain::Domain)::ResultSet

Create interface to a given Zarr result set.
"""
function load_results(result_loc::String)::ResultSet
    raw_set = zopen(joinpath(result_loc, RESULTS), fill_as_missing=false)
    log_set = zopen(joinpath(result_loc, LOG_GRP), fill_as_missing=false)
    input_set = zopen(joinpath(result_loc, INPUTS), fill_as_missing=false)

    r_vers_id = input_set.attrs["ADRIA_VERSION"]
    t_vers_id = "v" * string(PkgVersion.Version(@__MODULE__))

    if r_vers_id != t_vers_id
        msg = """Results were produced with an older version of ADRIA ($(r_vers_id)). The installed version of ADRIA is: $(t_vers_id)).\n
        Errors may occur when analyzing data."""

        @warn msg
    end

    input_cols::Array{String} = input_set.attrs["columns"]
    inputs_used::DataFrame = DataFrame(input_set[:, :], input_cols)

    env_layer_md::EnvLayer = EnvLayer(
        input_set.attrs["site_data_file"],
        input_set.attrs["site_id_col"],
        input_set.attrs["unique_site_id_col"],
        input_set.attrs["init_coral_cover_file"],
        input_set.attrs["connectivity_file"],
        input_set.attrs["DHW_file"],
        input_set.attrs["wave_file"]
    )

    return ResultSet(input_set.attrs["name"],
                     input_set.attrs["rcp"],
                     input_set.attrs["invoke_time"],
                     input_set.attrs["ADRIA_VERSION"],
                     input_set.attrs["site_ids"],
                     convert.(Float64, input_set.attrs["site_area"]),
                     convert.(Float64, input_set.attrs["site_max_coral_cover"]),
                     env_layer_md,
                     inputs_used,
                     input_set.attrs["sim_constants"],
                     NamedDimsArray{Symbol.(Tuple(raw_set.attrs["structure"]))}(raw_set),
                     NamedDimsArray{Symbol.(Tuple(log_set["rankings"].attrs["structure"]))}(log_set["rankings"]),
                     NamedDimsArray{Symbol.(Tuple(log_set["seed"].attrs["structure"]))}(log_set["seed"]),
                     NamedDimsArray{Symbol.(Tuple(log_set["fog"].attrs["structure"]))}(log_set["fog"]),
                     NamedDimsArray{Symbol.(Tuple(log_set["shade"].attrs["structure"]))}(log_set["shade"]))
end
function load_results(domain::Domain)::ResultSet
    log_location = joinpath(ENV["ADRIA_OUTPUT_DIR"], "$(domain.name)__$(domain.rcp)__$(domain.scenario_invoke_time)")
    return load_results(log_location)
end


"""
Hacky scenario filtering - to be replaced with more robust approach.

Only supports filtering by single attribute.
Should be expanded to support filtering metric results too.

# Examples
```julia
select(result, "guided .> 0.0")

# Above expands to:
# result.inputs.guided .> 0.0
```
"""
function select(r::ResultSet, op::String)
    scens = r.inputs

    col, qry = split(op, " ", limit=2)
    col = Symbol(col)

    df_ss = getproperty(scens, col)

    return eval(Meta.parse("$df_ss $qry"))
end


function Base.show(io::IO, mime::MIME"text/plain", rs::ResultSet)

    vers_id = rs.ADRIA_VERSION

    tf, species, sites, reps, scens = size(rs.raw)

    println("""
    Domain: $(rs.name)

    Run with ADRIA $(vers_id) on $(rs.invoke_time) for RCP $(rs.rcp)
    Results stored at: $(store_location(rs))

    Intervention scenarios run: $(scens)
    Environmental scenarios: $(reps)
    Number of sites: $(sites)
    Species/size groups represented: $(species)
    Timesteps: $(tf)

    Input layers
    ------------""")

    for fn in fieldnames(typeof(rs.env_layer_md))
        println("$(fn) : $(getfield(rs.env_layer_md, fn))")
    end
end
