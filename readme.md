# MQTT Broker w Mosquitto

## MQTT Broker functionalities:

- Publishing and Subscribing                        [receiver.py]
    - Heartbeat
    - Last Will and Testament
    - Image Capture
    - Commands

- Heartbeat Messages between Broker and Client      [receiver.py]
- Control Client through Broker Comms               [sendcommand.py]

- Simple scripts for sending b64 images over MQTT   [simple_image_receiver_b64.py]

## What it does

- [receiver.py] sets up the MQTT broker to subscribe/publish topics/messages over MQTT.
- [receiver.py] also establishes heartbeat messages for both broker and client on connect.
- [receiver.py] also establishes last will and testament for both broker and client.

- [sendcommand.py] allows the broker to communicate to client to do a specified command. 
    - For this example, is taking photo from client side and sending images back to broker.

- Images are stored in directories based on timestamp and client devices (cam).
- Logging is also done via console and and also store as .log files as well as CSV.