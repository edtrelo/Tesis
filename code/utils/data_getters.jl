using DataFrames
using CSV
using HTTP
using LibPQ
using Dates
using DotEnv

script_dir = @__DIR__  
env_path = joinpath(script_dir, "..", ".env")
DotEnv.load!(env_path, override = true);

host = ENV["SUPABASE_HOST"]
user = ENV["SUPABASE_USER"]
password = ENV["SUPABASE_PASSWORD"]
port = ENV["SUPABASE_PORT"]
dbname = ENV["SUPABASE_DB"]

"Obtiene un dataframe de un archivo csv alojado en una nube."
function read_csv_from_url(url::String)
    resp = HTTP.get(url)
    return CSV.read(IOBuffer(resp.body), DataFrame, types=Dict(:cve_ent => String))
end

"Transforma en DataFrame el csv crudo de infectados nuevos por día por municipio 
proporcionado por el CONACYT."
function get_COVIDMX_csv()
    url = "https://hcllgmdkkfcegvyizxqg.supabase.co/storage/v1/object/public/raw_src/Casos_Diarios_Municipio_Confirmados_20230625.csv"
    df = read_csv_from_url(url)
    return df
end

function get_new_infections_COVID_by_state(alias_ent::String)
    conn = LibPQ.Connection(
        "host=$host dbname=$dbname user=$user password=$password port=$port"
    );
    query = """
        select fecha, sum(casos) as casos from
        staging.new_infections a left join
        raw.ageeml b
        on a.cve_ent = b.mapa 
        where a.fuente = 'covid19mx'
        and b.alias_ent = '$alias_ent'
        group by fecha; 
    """
    df = DataFrame(execute(conn, query))
    close(conn)
    df.fecha = Date.(df.fecha, "dd-mm-yyyy")
    sort!(df, :fecha)
    return df
end

function get_new_infections_COVID_by_city(nom_mun::String)
    conn = LibPQ.Connection(
        "host=$host dbname=$dbname user=$user password=$password port=$port"
    );
    query = """
        select a.fecha, a.casos from
        staging.new_infections a left join
        raw.ageeml b
        on a.cve_ent = b.mapa 
        where a.fuente = 'covid19mx'
        and b.nom_mun = '$nom_mun'; 
    """
    df = DataFrame(execute(conn, query))
    close(conn)
    df.fecha = Date.(df.fecha, "dd-mm-yyyy")
    sort!(df, :fecha)
    return df
end

function get_population_ent(alias_ent::String)
    conn = LibPQ.Connection(
        "host=$host dbname=$dbname user=$user password=$password port=$port"
    );
    query = """
        select sum(poblacion) as poblacion from
        core.poblacion_prepandemia 
        where alias_ent =  '$alias_ent'
    """
    df = DataFrame(execute(conn, query))
    close(conn)
    return df.poblacion[1]
end

function get_population_mun(nom_mun::String)
    conn = LibPQ.Connection(
        "host=$host dbname=$dbname user=$user password=$password port=$port"
    );
    query = """
        select poblacion from
        core.poblacion_prepandemia 
        where nom_mun=  '$nom_mun'
    """
    df = DataFrame(execute(conn, query))
    close(conn)
    return df.poblacion[1]
end

df_I = get_new_infections_COVID_by_state("Quintana Roo")
using Plots
plot(df_I.casos)

get_population_ent("Ciudad de México")