import arrow                        
import db_helper as db
import helper
import process_img_rpi as pir

def insert_db(value):  
    """
        Sets Up DB on PSQL, Runs the image processing algorithm, Store in DB
    """

    try:
        try:
            print("Starting Program")

            # Load adminConfigs
            main_config, db_config = helper.get_config()
            spt_db_connections = db_config["spt_db_connections"]

            # Start Logger
            logger = helper.start_logger(main_config)

            # Retrieve Config
            logger.info("Retrieving line config")
            line_config = main_config["line_config"]

            # Base Folder
            output_path = helper.create_base_folder(main_config)
            logger.info("Created Base Folder")

            # Retrieve Base Information on Host (RPI)
            hostname, fqdn, local_ip = helper.get_host_info()
            program_mode = main_config["program_mode"]
            program_version = main_config["program_version"]
            run_date = arrow.now().format("YYYY-MM-DD")

            for line_key, line_value in line_config.items():
                logger.info("Running Line: %s",line_key)
                
                cam = line_value["Cam"]
                ip_addr = line_value["IP"]
                logger.info("Running Pump: %s inserted into cryro_pump_uat.cryro_pump_pressure_header", line_key)
                logger.info("Cam: %s", cam)
                logger.info("IP: %s", ip_addr)
                
                params = (
                    line_key,
                    run_date,
                    hostname,
                    fqdn,
                    local_ip,
                    program_mode,
                    program_version,
                    cam,
                    ip_addr
                )

                # Does the inserting of values and returns the pressure id
                run_id = db.create_pressure_header_main(params, spt_db_connections)

            # Insert into DB
            try:
                spec_id = 1
                status = "PASS"
                status_msg = "PASS"
                result = value
                run_data_param = (
                    run_id,
                    spec_id,
                    status,
                    status_msg,
                    result,
                    output_path
                )
                db.create_run_data_main(run_data_param, spt_db_connections)
            except Exception as e:
                raise e
                
        except Exception as e:
            print(e)
            logger.error(e)
            status = "FAIL"
            status_msg = e
            run_data_param = (
                run_id,
                spec_id,
                status,
                status_msg,
                result,
                output_path
            )

            logger.error(run_data_param)
            db.create_run_data_main(run_data_param, spt_db_connections)
            logger.error("Error Completed")
    
    except Exception as e:
        pass

def main():
    """ Run insert_ db with dummy value """
    insert_db(1.5)
        
if __name__ == "__main__":
    main()
