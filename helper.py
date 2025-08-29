"""
    Helper functions for the Database APIs in db_helper.py
"""

import json
import os
import logging
from logging.handlers import RotatingFileHandler
import socket
import time
import arrow

logs = logging.getLogger("PRESSURE")

def get_config():

    """ Get the paths from config.json and db_connection.json and load them """

    try:
        main_config_path = open(
            "/home/admin/Desktop/pressure_gauge_subscriber/Config/config.json", encoding="utf-8"
        )

        main_config = json.load(main_config_path)

        db_config_path = open(
            "/home/admin/Desktop/pressure_gauge_subscriber/Sensitive/db_connection.json",  encoding="utf-8"
        )

        db_config = json.load(db_config_path)

        return main_config, db_config
    
    except Exception as e:
        raise e
     
def start_logger(main_config):

    """ Sets up the logger based on specified format """
    
    try:
        log_base_fp = main_config["log_base_fp"]
        logpath = main_config["log_fp"]

        if not os.path.exists(log_base_fp):
            os.makedirs(log_base_fp)
            time.sleep(0.5)

        logger = logging.getLogger("PRESSURE")

        if logger.hasHandlers():
            logger.handlers.clear()

        handler = RotatingFileHandler(logpath, maxBytes=2000000, backupCount=30)

        logger.setLevel(logging.DEBUG)

        handler.suffix = "%Y%m%d"
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.info("Starting Process")
        logger.info("Get Processing Time \t %s", arrow.now().format('YYYY-MM-DD_HH-mm-ss'))

        return logger
    
    except Exception as e:
        raise e

def create_base_folder(main_config):

    """ Creates base directory based on the current time """
    try:
        dt_obj = arrow.now().format("YYYY_MM_DD_HH_mm_ss")
        base_output_path = main_config["base_output_path"]
        output_path = base_output_path + "/" + dt_obj

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        return output_path
    
    except Exception as e:
        raise e
    
def get_host_info():

    """ Retrieves the hostname, fqdn, local_ip and raise Exception when error """

    try:
        logs.info("Retrieving Host Info Started")
        hostname = ""
        fqdn = ""
        local_ip = ""

        hostname = socket.gethostname()
        fqdn = socket.getfqdn()
        local_ip = socket.gethostbyname(fqdn)

        return hostname, fqdn, local_ip
    
    except ValueError as e:
        return e