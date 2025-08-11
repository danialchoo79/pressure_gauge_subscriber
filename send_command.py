import sys
import time
import logging
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "128.53.209.104"
MQTT_PORT = 1883

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def send_command(camera_id, command):
    
    topic = f"cameras/{camera_id}/cmd"
    client = mqtt.Client()

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"Connected to broker. Sending {command} to {topic}")
            client.publish(topic, command, qos=1)
            client.disconnect()
        else:
            logging.error(f"Failed to connect to MQTT Broker, rc={rc}")

    
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    time.sleep(2)
    client.loop_stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python send_command.py <camera_id> <command>")
        sys.exit(1)

    cam_id = sys.argv[1]
    cmd = sys.argv[2]
    send_command(cam_id, cmd)
