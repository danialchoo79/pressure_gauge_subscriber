import paho.mqtt.client as mqtt
import os
import csv
import datetime
import threading
import logging
import time

# MQTT Configuration
MQTT_BROKER = "128.53.209.104"
MQTT_PORT = 1883
MQTT_TOPIC = "cameras/+/image"
HEARTBEAT_TOPIC = "system/heartbeat"
CLIENT_ID = "raspi_receiver"

# Base Directory to Saved Images
SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# CSV Log File
MASTER_LOG_FILE = os.path.join(SAVE_DIR, "image_log.csv")
if not os.path.exists(MASTER_LOG_FILE):
    with open(MASTER_LOG_FILE, mode="w", newline="") as log_csv:
        writer = csv.writer(log_csv)
        writer.writerow(["Timestamp","Camera","Saved File","Topic","Size (Bytes)"])

# Logging in Directory
logging.basicConfig(
    filename=os.path.join(SAVE_DIR,"broker.log"),
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Logging in Console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# MQTT Client
client = mqtt.Client(client_id=CLIENT_ID)
client.will_set("system/status", payload="raspi disconnected", qos=1, retain=True)

heartbeat_timer = None
message_count = 0

# === Utility Functions ===

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_date_folder():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def send_heartbeat():
    global heartbeat_timer
    try:
        client.publish(HEARTBEAT_TOPIC, payload="alive", qos=0, retain=False)
        logging.info(f"{CLIENT_ID} : Heartbeat Sent.")

    except Exception as e:
        logging.error(f"Heartbeat failed: {e}")

    finally:
        heartbeat_timer = threading.Timer(30.0, send_heartbeat) # every 30s
        heartbeat_timer.start()

def cancel_heartbeat():
    global heartbeat_timer
    
    if heartbeat_timer is not None:
        heartbeat_timer.cancel()  
        logging.info("Heartbeat Cancelled.")
        heartbeat_timer = None

def get_cam_log_path(camera_id):
    return os.path.join(SAVE_DIR, f"{camera_id}_log.csv")

# === MQTT Callbacks ===

def on_connect(client, userdata, flags, rc):
    if rc == 0:

        # Images
        logging.info(f"Connected to MQTT broker with result code: {rc}")
        client.subscribe(MQTT_TOPIC, qos=1)
        logging.info(f"Subscribed to topic: {MQTT_TOPIC}")

        # Heartbeat Messages
        client.subscribe(f"{HEARTBEAT_TOPIC}/+",qos=0)
        
        logging.info(f"Subscribed to topic: {HEARTBEAT_TOPIC}")

        send_heartbeat() # Start heartbeat after connection
    else:
        logging.error(f"Failed to connect to MQTT Broker, result code: {rc}")
                    
def on_message(client, userdata, msg):
    global message_count

    if(msg.topic.startswith("system/heartbeat/")):
        sender_id = msg.topic.split('/')[-1] # Get camera ID from Topic
        logging.info(f"{sender_id} : Heartbeat Received.")

        # Ignore own Heartbeat
        if sender_id == CLIENT_ID:
            return
        return

    try:
        image_data = msg.payload # raw binary

        if not image_data:
            logging.warning("Warning - Empty Payload Received")
            return
        
        # Extract camera ID from Topic
        topic_parts = msg.topic.split("/")
        camera_id = topic_parts[1] if len(topic_parts) >= 2 else "unknown_cam"

        #Create folder for today
        date_folder = get_date_folder()
        save_path = os.path.join(SAVE_DIR, date_folder, camera_id)
        os.makedirs(save_path, exist_ok=True)

        # Timestamped filename
        timestamp = get_timestamp()
        filename = f"{camera_id}_{timestamp}.jpg"
        full_path = os.path.join(save_path, filename)

        # Save image
        with open(full_path, "wb") as f:
            f.write(image_data)
        
        logging.info(f"Image saved: {full_path} with {len(image_data)} bytes")

        # Per-camera CSV Log File
        per_cam_log_file_path = get_cam_log_path(camera_id)

        # Per-camera CSV Log File [Headers]
        if not os.path.exists(per_cam_log_file_path):
            with open(per_cam_log_file_path, mode="w", newline="") as cam_log:
                writer = csv.writer(cam_log)
                writer.writerow(["Timestamp","Saved File", "Topic", "Size (Bytes)"])

        # Append to Per-Camera CSV Log
        with open(per_cam_log_file_path, mode="a", newline="") as cam_log:
            writer = csv.writer(cam_log)
            writer.writerow([timestamp, full_path, msg.topic, len(image_data)])

        # Append to MASTER CSV Log
        with open(MASTER_LOG_FILE, mode="a", newline="") as log_csv:
            writer = csv.writer(log_csv)
            writer.writerow([timestamp, camera_id, full_path, msg.topic, len(image_data)])

        message_count += 1
        logging.info(f"Total messages received: {message_count}")

    except Exception as e:
        logging.error(f"Failed to handle incoming image: {e}")

# Disconnect
def on_disconnect(client, userdata, rc):
    cancel_heartbeat()
    logging.warning("Disconnected from broker. Trying to reconnect...")
    while rc != 0:
        try:
            rc = client.reconnect()
            if rc == 0:
                logging.info("Reconnected to broker successfully.")
                client.subscribe(MQTT_TOPIC, qos=1)
                logging.info(f"Resubscribed to topic: {MQTT_TOPIC}")
                send_heartbeat()
        except Exception as e:
            logging.error(f"Reconnect failed: {e}")
            time.sleep(5)

# === Setup and Start ===

client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

try:
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    logging.info(f"Listening for MQTT messages on {MQTT_TOPIC}")

    try:
        client.loop_forever()

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting...")
        cancel_heartbeat()
        client.disconnect()

except Exception as e:
    logging.critical(f"Could not start MQTT Client: {e}")