import depthai as dai

from robothub_sdk import App, StreamType, Config
from robothub_sdk.device import Device

class Sensor():
    _str_to_sensor = {
            "accelerometer": dai.IMUSensor.ACCELEROMETER_RAW,
            "magnetometer": dai.IMUSensor.MAGNETOMETER_RAW,
            "gyroscope": dai.IMUSensor.GYROSCOPE_RAW,
            "rotation": dai.IMUSensor.ROTATION_VECTOR
    }
    
    def str_to_sensor(str_sensor):
        return Sensor._str_to_sensor[str_sensor]

    def get_data_from_packet(packet: dai.IMUPacket, sensors):
        data = {}
        if dai.IMUSensor.ACCELEROMETER_RAW in sensors:
            accel = packet.acceleroMeter
            data["accelerometer"] = {"x": accel.x, "y": accel.y, "z": accel.z}
        
        if dai.IMUSensor.GYROSCOPE_RAW in sensors:
            gyro = packet.gyroscope
            data["gyroscope"] = {"x": gyro.x, "y": gyro.y, "z": gyro.z}

        if dai.IMUSensor.MAGNETOMETER_RAW in sensors:
            magn = packet.magneticField
            data["magnetometer"] = {"x": magn.x, "y": magn.y, "z": magn.z}
        
        if dai.IMUSensor.ROTATION_VECTOR in sensors:
            rot = packet.rotationVector
            data["rotation"] = {"i": rot.i, "j": rot.j, "k": rot.k, "real": rot.real}

        return data

    def get_sensors(sensor_list):
        return list(map(Sensor.str_to_sensor, sensor_list))

class IMUApp(App):

    def on_initialize(self, unused_devices):
        print("Initializing app ...")

        self.last_secs = 0
        self.default_sensors = Sensor.get_sensors(["accelerometer", "gyroscope", "magnetometer", "rotation"])
        self.default_rate = 1

    def on_configuration(self, old_configuration: Config):
        print("Configuration update", self.config.values())
        
        if self.config.rate != old_configuration.rate:
            self.rate = self.config.rate
        
        if self.config.sensors != old_configuration.sensors:
            self.sensors = Sensor.get_sensors(self.config.sensors)
            self.restart()

    def on_setup(self, device: Device) -> None:
        print("Setting the device up ...")

        self.sensors = Sensor.get_sensors(self.config.sensors) if self.config.sensors else self.default_sensors
        self.rate = self.config.rate if self.config.rate else self.default_rate

        RATE = 50 # NOTE: default
        STREAM_RATE = 1

        imu = device.create_imu(self.sensors, RATE)
        imu_stream = device.streams.create(imu, imu.out, StreamType.IMU, rate=STREAM_RATE)
        imu_stream.consume(self.print_imu)

    def print_imu(self, data: dai.IMUData) -> None:
        for packet in data.packets:
            secs = packet.acceleroMeter.timestamp.get().total_seconds()
            if secs - self.last_secs >= 1/self.rate:
                self.last_secs = secs
                data = Sensor.get_data_from_packet(packet, self.sensors)
                print(data)

app = IMUApp()
app.run()
