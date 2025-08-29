import arrow                        
import db_helper as db
import helper
import process_img_rpi as pir

def main():  
    """
        Sets Up DB on PSQL, Runs the image processing algorithm, Store in DB
    """

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

        # Retrieve Base Information on Host
        hostname, fqdn, local_ip = helper.get_host_info()
        program_mode = main_config["program_mode"]
        program_version = main_config["program_version"]
        run_date = arrow.now().format("YYYY-MM-DD")

        for line_key, line_value in line_config.items():
            logger.info("Running Line: %s",line_key)

            params = (
                line_key,
                run_date,
                hostname,
                fqdn,
                local_ip,
                program_mode,
                program_version
            )

            run_id = db.create_pressure_header_main(params, spt_db_connections)

            for pump_key, pump_value in line_value.items():

                ip_addr = pump_value["IP"]
                cam = pump_value["Cam"]
                logger.info("Running Pump: %s", pump_key)
                logger.info("IP: %s", ip_addr)
                logger.info("Cam: %s", cam)

                # Init
                spec_id = 1
                status = ""
                status_msg = ""
                result = ""

                try:
                    # run image processing
                    pass
                except Exception as e:
                    logger.error(e)
                    raise ValueError("Unable to generate result from image") from e
                
                # Insert into DB
                try:
                    spec_id = 1
                    status = "PASS"
                    status_msg = "PASS"
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
            
    except ValueError as e:
        pass

if __name__ == "__main__":
    main()
