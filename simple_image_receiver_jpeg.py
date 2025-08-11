import paho.mqtt.client as mqtt
import os

MQTT_BROKER = "128.53.209.104"
MQTT_PORT = 1883
MQTT_TOPIC = "cameras/cam1/image"

SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        image_data = msg.payload # raw binary

        filename = os.path.join(SAVE_DIR, f"cam1_latest.jpg")

        with open(filename, "wb") as f:
            f.write(image_data)
        
        print(f"Image saved: {filename}")

    except Exception as e:
        print("Failed to handle incoming image:",e)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
print(f"Listening for MQTT messages on {MQTT_TOPIC}...")

client.loop_forever()