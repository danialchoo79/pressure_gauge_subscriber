import psycopg2

# Connections and Selecting of data items in specific tables of PGSQL Database.

def getpgconnection(sqlparam):

    """ Param: 1d Array [database name, username, password, host]
        Returns: psycopg2 connection obj
    """

    try:
        conn = psycopg2.connect(
            dbname=sqlparam[0],
            user=sqlparam[1],
            password=sqlparam[2],
            host=sqlparam[3]
        )

        return conn

    except Exception as e:
        raise e
    
def create_pressure_header(pgConn, data):
    
    """
        Selects first 7 items from the first row from cryro_pump_pressure_header
            - [run_hdr_id, line, chamber, run_date, created_dt, host_name, local_ip]
    """
    
    try:
        cur = pgConn.cursor()
        cur.execute("SELECT * FROM cryro_pump_pressure_header(%s,%s,%s,%s,%s,%s,%s);", data)
        result = cur.fetchall()[0]
        run_id = result[0]
        error_code = result[1]

        if(error_code == '1'):
            raise ValueError("Error create_pressure_header due to - ", result[2])
        
        pgConn.commit()
        return run_id
    
    except Exception as e:
        raise e
    
def create_pressure_header_main(params, spt_db_connections):

    """
        Connects to pgsql then selects first 7 items from the first \
        row from cryro_pump_pressure_header

    """
    try:
        pgConn = None
        pgConn = getpgconnection((spt_db_connections['DATABASE'],
                                  spt_db_connections['USER'],
                                  spt_db_connections['PASSWORD'],
                                  spt_db_connections['HOST']
        ))

        run_id = create_pressure_header(pgConn, params)
        return run_id
    
    except Exception as e:
        raise e

    finally:
        if(pgConn is not None):
            pgConn.close()

def create_run_data(pgConn,data):

    """ Selects the first row from create_cryro_pump_run_data """

    try:
        cur = pgConn.cursor()
        cur.execute("SELECT * FROM cryro_pump_uat.create_cryro_pump_run_data(%s,%s,%s,%s,%s,%s);", data)
        result = cur.fetchall()[0]
        error_code = result[0]

        if(error_code == "1"):
            raise ValueError(f"Error create_run_data due to - ,{result[1]}")
        pgConn.commit()

    except Exception as e:
        raise e

def create_run_data_main(params, spt_db_connections):

    """ Connects to PGSQL and selects the first row from create_cryro_pump_run_data """

    try:
        pgConn = None
        pgConn = getpgconnection((spt_db_connections['DATABASE'],
                                    spt_db_connections['USER'],
                                    spt_db_connections['PASSWORD'],
                                    spt_db_connections['HOST']
        ))

        create_run_data(pgConn, params)
    
    except Exception as e:
        raise e
    
    finally:
        if(pgConn is not None):
            pgConn.close()