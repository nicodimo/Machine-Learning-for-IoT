import json
import uuid
from time import sleep, time

import paho.mqtt.client as mqtt
import psutil


class MqttClient:
    def __init__(self, broker: str, port: int, keepalive: int = 60):
        self.broker = broker
        self.port = port
        self.keepalive = keepalive
        self.client = mqtt.Client()

    def connect(self):
        self.client.connect(host=self.broker, port=self.port, keepalive=self.keepalive)

    def publish(self, topic, message):
        self.client.publish(topic, message)


if __name__ == '__main__':

    broker = 'test.mosquitto.org'
    port = 1883
    topic = 's290453'

    client = MqttClient(broker, port)
    client.connect()
    mac_address = hex(uuid.getnode())

    while True:
        sleep(1)
        timestamp = time()
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)

        info = {
            "mac_address": mac_address,
            "timestamp": timestamp,
            "battery_level": battery_level,
            "power_plugged": power_plugged
        }

        message = json.dumps(info)
        client.publish(topic, message)
