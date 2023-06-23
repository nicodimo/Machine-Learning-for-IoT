from math import floor

import redis
import argparse
import psutil
import uuid
from datetime import datetime
from time import sleep
from time import time

REDIS_HOST = 'redis-11747.c300.eu-central-1-1.ec2.cloud.redislabs.com'
REDIS_PORT = 11747
REDIS_PASSWORD = 'i4Wds8wGEcQ5PVQwBMmEQL17GH4XdXKH'  # i4Wds8wGEcQ5PVQwBMmEQL17GH4XdXKH

PLAIN_MEMORY_LIMIT = 5 * 1024 * 1024  # Memory limit for "battery" and "power" timeseries
AGGREGATED_MEMORY_LIMIT = 1024 * 1024  # Memory limit for "plugged" timeseries

# 86400 day seconds


def initialize(hostname: str = REDIS_HOST, port: int = REDIS_PORT, password: str = REDIS_PASSWORD) -> redis.Redis:
    r = None
    try:
        r = redis.Redis(host=hostname, port=port, password=password)
        is_connected = r.ping()
        print('Redis client is connected ', is_connected)
    except Exception as ex:
        print(ex)
    return r


def init_timeseries(db: redis.Redis, address: str):
    db.delete(f'{address}:battery')
    db.delete(f'{address}:power')
    db.delete(f'{address}:plugged')

    # find battery timeseries retention time by checking the memory occupied by a single value.
    battery_memory_usage = 16
    battery_retention = floor(PLAIN_MEMORY_LIMIT / (battery_memory_usage * 0.10)) * 1000
    print(f'The memory used to store one value of the battery level timeseries is {battery_memory_usage}, the retention time: {str(battery_retention)} msec.')

    # find power timeseries retention time by checking the memory occupied by a single value.
    power_memory_usage = 16
    power_retention = floor(PLAIN_MEMORY_LIMIT / (power_memory_usage * 0.10)) * 1000
    print(f'The memory used to store one value of the power level timeseries is {power_memory_usage}, the retention time: {str(power_retention)} msec.')

    # find plugged timeseries retention time by checking the memory occupied by a single value.
    plugged_memory_usage = 16
    plugged_retention = floor(AGGREGATED_MEMORY_LIMIT / (plugged_memory_usage * 0.10)) * 86400 * 1000
    print(
        f'The memory used to store one value of the power level timeseries is {plugged_memory_usage}, the retention time: {str(plugged_retention)} msec.')

    db.ts().create(f'{address}:battery', retention_msecs=battery_retention)
    db.ts().create(f'{address}:power', retention_msecs=power_retention)
    db.ts().create(f'{address}:plugged', retention_msec=plugged_retention)

    t = db.ts().info(f'{address}:power')

    db.ts().createrule(source_key=f'{address}:power', dest_key=f'{address}:plugged', aggregation_type='sum', bucket_size_msec=86400)  # 86400


def check_memory_usage(db: redis.Redis, address: str):
    battery_memory_usage = db.memory_usage(f'{address}:battery')
    power_memory_usage = db.memory_usage(f'{address}:power')
    plugged_memory_usage = db.memory_usage(f'{address}:plugged')

    print(f'Memory usage: ')
    print(f'{address}:battery -> {battery_memory_usage}')
    print(f'{address}:power -> {power_memory_usage}')
    print(f'{address}:plugged -> {plugged_memory_usage}')

    if battery_memory_usage >= PLAIN_MEMORY_LIMIT:
        print('Battery timeseries violated memory limit.')
    if power_memory_usage >= PLAIN_MEMORY_LIMIT:
        print('Power timeseries violated memory limit.')
    if plugged_memory_usage >= AGGREGATED_MEMORY_LIMIT:
        print('Plugged timeseries violated memory limit')


def save_data(db: redis.Redis, address: str, battery, power, tmp):
    db.ts().add(f'{address}:battery', int(tmp), battery)
    db.ts().add(f'{address}:power', int(tmp), power)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Example launch command "python lab1_ex2.py --hostname xxxx --port xxxxx --password xxxx')
    parser.add_argument('--hostname', nargs='?', help='Redis database hostname')
    parser.add_argument('--port', nargs='?', help='Redis database port')
    parser.add_argument('--password', nargs='?', help='Redis database password')

    args = parser.parse_args()
    hostname = args.hostname if args.hostname is not None else REDIS_HOST
    password = args.password if args.password is not None else REDIS_PASSWORD
    port = args.port if args.port is not None else REDIS_PORT

    client = initialize(hostname=hostname, password=password, port=port)
    if client is None:
        exit(1)

    mac_address = hex(uuid.getnode())
    init_timeseries(client, mac_address)
    i = 0
    while True:
        if i % 100 == 0:
            check_memory_usage(client, mac_address)
        timestamp = time()
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)

        formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f'{formatted_datetime} {mac_address} : battery : {battery_level}%')
        print(f'{formatted_datetime} {mac_address} : power = {power_plugged}')
        save_data(client, mac_address, battery_level, power_plugged, datetime.now().timestamp())
        sleep(1)
        i += 1



