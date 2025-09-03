# pylint: disable=all

import paho.mqtt.client as mqtt
import os
import csv
import datetime
import time
import random
import threading
import logging
import time
from process_img_rpi import main as process_image_main
import subprocess
import queue
import read_pressure_main as rpm

# MQTT Configuration
MQTT_BROKER = "128.53.209.104"
MQTT_PORT = 1883

# Topics
CAMERA_TOPIC = "cameras/+/image"
HEARTBEAT_TOPIC = "system/heartbeat"
COMMAND_TOPIC = "cameras/+/cmd"
BROKER_ID = "raspi_receiver"
LWT_CLIENT_TOPIC = "system/status/lwt/+"
LWT_BROKER_TOPIC = f"system/status/lwt/{BROKER_ID}"

# Base Directory for Saved Images
SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Master CSV log
MASTER_LOG_FILE = os.path.join(SAVE_DIR, "image_log.csv")
if not os.path.exists(MASTER_LOG_FILE):
    with open(MASTER_LOG_FILE, mode="w", newline="") as log_csv:
        writer = csv.writer(log_csv)
        writer.writerow(["Timestamp", "Camera", "Saved File", "Topic", "Size (Bytes)"])

# Logging Setup
logging.basicConfig(
    filename=os.path.join(SAVE_DIR, "broker.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# MQTT Client
client = mqtt.Client(client_id = BROKER_ID)
client.will_set("system/status/raspi", payload="disconnected", qos=1, retain=True)

# Heartbeat and Counter
heartbeat_timer = None
message_count = 0

# Queue
image_queue = queue.Queue()
result_queue = queue.Queue()

# === Utility Functions ===

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_date_folder():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def get_cam_log_path(camera_id):
    return os.path.join(SAVE_DIR, f"{camera_id}_log.csv")

def send_heartbeat():
    global heartbeat_timer
    try:
        client.publish(f"{HEARTBEAT_TOPIC}/{BROKER_ID}", payload=BROKER_ID, qos=0)
        logging.info(f"BROKER_ID ({BROKER_ID}) : Heartbeat sent.")
    except Exception as e:
        logging.error(f"Heartbeat failed: {e}")
    finally:
        heartbeat_timer = threading.Timer(30.0, send_heartbeat)
        heartbeat_timer.start()

def cancel_heartbeat():
    global heartbeat_timer
    if heartbeat_timer is not None:
        heartbeat_timer.cancel()
        logging.info("Heartbeat cancelled.")
        heartbeat_timer = None

def run_processing(full_path, output_dir):
    """ Runs the image processing script"""

    result = subprocess.run(
        ["python3","process_img_rpi.py", full_path, output_dir,"lineA","pumpA"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        try:
            value = float(result.stdout.strip())
            logging.info(f"Processing result for {full_path} : {value}")
            return value
        except ValueError:
            logging.error(f"Unexpected Output: {result.stdout}")
            return None
        
    else:
        logging.error(f"Process failed: {result.stderr}")
        return None
            
def process_worker():
    """ Continuously process images from the queue """
    while True:
        full_path, output_dir = image_queue.get()
        value = run_processing(full_path, output_dir)
        
        if value is not None:
            logging.info(f"Value is {value}")
            result_queue.put((full_path, value))
       
        image_queue.task_done()

def result_handler():
    while True:
        full_path, value = result_queue.get()
        logging.info(f"Handling result for {full_path} = {value}")
        # Store the value in the Database
        rpm.insert_db(value)
        result_queue.task_done()

# Start 4 Image Process workers in the background
for _ in range(4):
    threading.Thread(target=process_worker, daemon=True).start()

# Start 1 DB Process worker in the background
threading.Thread(target=result_handler, daemon = True).start()

# === MQTT Callbacks ===
def on_connect(client, userdata,flags, rc):
    if rc == 0:
        logging.info(f"Connected to MQTT Broker with result code: {rc}")
        client.subscribe(CAMERA_TOPIC, qos=1)
        client.subscribe(f"{HEARTBEAT_TOPIC}/+", qos=0)
        client.subscribe(LWT_CLIENT_TOPIC, qos=1)
        client.subscribe(COMMAND_TOPIC, qos=1)
        logging.info(f"All subscriptions done. Clients subscribed: {CAMERA_TOPIC}, {LWT_CLIENT_TOPIC}, {COMMAND_TOPIC}")
        send_heartbeat()
    else:
        logging.error(f"Failed to connect to MQTT Broker, rc={rc}")

def on_message(client, userdata, msg):
    global message_count

    payload_str = msg.payload.decode("utf-8", errors="ignore")

    # LWT
    if msg.topic.startswith("system/status/lwt/"):
        camera_id = msg.topic.split("/")[3]
        logging.warning(f"LWT from {camera_id}: {payload_str}")
        return
    
    # Heartbeat
    if msg.topic.startswith("system/heartbeat/"):
        sender_id = msg.topic.split("/")[-1]
        logging.info(f"HEARTBEAT_ID ({sender_id}) : Heartbeat received.")
        if sender_id != BROKER_ID:
            return
        return
    
    # Command
    if msg.topic.startswith("cameras/") and msg.topic.endswith("/cmd"):
        camera_id = msg.topic.split("/")[1]
        logging.info(f"Command for {camera_id} : {payload_str}")
        return
    
    # Image
    if msg.topic.startswith("cameras/") and msg.topic.endswith("/image"):

        try:
            image_data = msg.payload
            
            if not image_data:
                logging.warning("Empty image payload.")
                return
            
            camera_id = msg.topic.split("/")[1]
            date_folder = get_date_folder()
            save_path = os.path.join(SAVE_DIR, date_folder, camera_id)

            # Ensure Save Path Exists
            try:
                os.makedirs(save_path, exist_ok=True)
            except Exception as e:
                logging.error(f"Failed to create directory {save_path} : {e}")
                return

            # Save Image
            timestamp = get_timestamp()
            filename = f"{camera_id}_{timestamp}.jpg"
            full_path = os.path.join(save_path, filename)

            try:
                with open(full_path, "wb") as f:
                    f.write(image_data)
                    logging.info(f"Image saved: {full_path}. Size: {len(image_data)} bytes.")

                    # === Do Image Processing ===
                    
                    output_dir = os.path.join(save_path, "processed")
                    os.makedirs(output_dir, exist_ok=True)

                    # enqueue the processing task
                    image_queue.put((full_path, output_dir))
                    logging.info(f"Queued image for processing: {full_path}")
                    return

            except Exception as e:
                logging.error(f"Image saved: {full_path} : {e}")
            
            # Per-Camera CSV Log
            per_cam_log_file = get_cam_log_path(camera_id)

            if not os.path.exists(per_cam_log_file):
                with open(per_cam_log_file, mode="w", newline="") as cam_log:
                    writer = csv.writer(cam_log)
                    writer.writerow(["Timestamp", "Saved File", "Topic", "Size (Bytes)"])
                
            with open(per_cam_log_file, mode="a", newline="") as cam_log:
                writer = csv.writer(cam_log)
                writer.writerow([timestamp, full_path, msg.topic, len(image_data)])

            # Master cSV
            with open(MASTER_LOG_FILE, mode="a", newline="") as log_csv:
                writer = csv.writer(log_csv)
                writer.writerow([timestamp, camera_id, full_path, msg.topic, len(image_data)])

            message_count += 1
            logging.info(f"Total images received: {message_count}")

        except Exception as e:
            logging.error(f"Image handling failed: {e}")

def on_disconnect(client, userdata, rc):
    
    cancel_heartbeat()
    logging.warning("Disconnected from broker. Reconnecting...")

    while True:
        try:
            rc = client.reconnect()

            if rc == 0:
                logging.info("Reconnected successfully.")
                client.subscribe(CAMERA_TOPIC, qos=1)
                send_heartbeat()
                break

        except Exception as e:
            logging.error(f"Recconect failed: {e}")
            time.sleep(5)

# Client Operations
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

# Main 
if __name__ == "__main__":
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_forever()
    except KeyboardInterrupt:
        cancel_heartbeat()
        client.disconnect()
    except Exception as e:
        logging.critical(f"Could not start MQTT Client: {e}")
