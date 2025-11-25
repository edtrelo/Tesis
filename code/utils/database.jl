using LibPQ, DataFrames, CSV
using DotEnv
script_dir = @__DIR__  
include(joinpath(script_dir, "data_getters.jl"))

env_path = joinpath(script_dir, "..", ".env")
DotEnv.load!(env_path, override = true);

host = ENV["SUPABASE_HOST"]
user = ENV["SUPABASE_USER"]
password = ENV["SUPABASE_PASSWORD"]
port = ENV["SUPABASE_PORT"]
dbname = ENV["SUPABASE_DB"]

function pg_type(el)
    if el <: Integer
        return "INTEGER"
    elseif el <: AbstractFloat
        return "FLOAT"
    elseif el <: AbstractString
        return "TEXT"
    elseif el <: Date
        return "DATE"
    else
        return "TEXT"  
    end
end

function create_raw_COVID_mex!()
    raw_data = get_full_new_infections_COVID()
    conn = LibPQ.Connection(
        "host=$host dbname=$dbname user=$user password=$password port=$port"
    );
    schema = "raw"
    table = "COVID19_new_infections_mex"

    # construir query automáticamente
    cols_defs = String[]
    for col in names(raw_data)
        push!(cols_defs, "\"$col\" $(pg_type(eltype(raw_data[!, col])))")
    end

    create_table_query = """
    CREATE TABLE IF NOT EXISTS $schema.$table (
        $(join(cols_defs, ",\n    "))
    );
    """;
end

function create_staging_new_infected!()
    conn = LibPQ.Connection(
        "host=$host dbname=$dbname user=$user password=$password port=$port"
    );
    raw_data = execute(conn, "SELECT * FROM raw.COVID19_new_infections_mex")
    raw_df = DataFrame(raw_data)
    select!(raw_df, Not(:poblacion))
    df_long = stack(
        raw_df, 
        Not([:cve_ent, :nombre]);
        variable_name = :fecha,
        value_name = :casos
    )
    df_long.database = fill("covid19mx", nrow(df_long))

    schema = "staging"
    table = "new_infections"

    create_table_query = """
    CREATE TABLE IF NOT EXISTS $schema.$table (
        cve_ent    TEXT,
        nombre     TEXT,
        fecha      TEXT,
        casos      INTEGER,
        fuente  TEXT
    );
    """;
    execute(conn, create_table_query)

    row_strings = map(eachrow(df_long)) do row
        # escapando valores por si acaso
        cve = row.cve_ent
        nombre = row.nombre
        fecha = string(row.fecha)  # Date → String en formato ISO
        casos = row.casos
        db = row.database
        "$cve,$nombre,$fecha,$casos,$db\n"
    end

    copyin = LibPQ.CopyIn("COPY staging.new_infections (cve_ent,nombre,fecha,casos,database) FROM STDIN (FORMAT CSV);", row_strings)
    execute(conn, copyin)
    close(conn) 
end

create_staging_new_infected!()
