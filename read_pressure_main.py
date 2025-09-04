import arrow                        
import db_helper as db
import helper
import process_img_rpi as pir

def insert_db(value, cam_name):  
    """
        Sets Up DB on PSQL, Runs the image processing algorithm, Store in DB
    """

    try:
        print("Starting Insert DB Program")

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

        # Loop lines -> pumps -> cams
        run_id = None
        ip_addr = None

        for line_key, pumps in line_config.items():
            for pump_key, cams in pumps.items():
                print("Looking for camera: ", cam_name, "in config: ", cams.keys())
                if cam_name in cams:

                    ip_addr = cams[cam_name]["ip"]

                    logger.info("Running Line: %s",line_key)
                    logger.info("Pump: %s", pump_key)
                    logger.info("Cam: %s", cam_name)
                    logger.info("IP: %s", ip_addr)
            
                    params = (
                        line_key,
                        run_date,
                        hostname,
                        fqdn,
                        local_ip,
                        program_mode,
                        program_version,
                        cam_name,
                        ip_addr
                    )

                    # Does the inserting of values and returns the pressure id
                    run_id = db.create_pressure_header_main(params, spt_db_connections)
                    logger.info("DB header created for %s with run_id=%s", cam_name, run_id)
                    print("run_id: ",run_id)
                    break
            if run_id:
                break
        
        if run_id is None:
            raise ValueError(f"Camera {cam_name} is not found in config.")

        spec_id = 1
        status = "PASS"
        status_msg = "PASS"
        result = value
        
        try:   
            run_data_param = (
                run_id,
                spec_id,
                status,
                status_msg,
                result,
                output_path
            )
            db.create_run_data_main(run_data_param, spt_db_connections)
            logger.info(f"Inserted run data for {cam_name}")

        except Exception as e:
            print("Insert DB failed: ", e)
            if 'logger' in locals():
                logger.error(f"Insert DB failed {e}")
            
    except Exception as e:
        print(e)
        logger.error("Error: %s",str(e))
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

def main():
    """ Run insert_ db with dummy value """
    insert_db(1.5, "dummy_cam")
        
if __name__ == "__main__":
    main()
