"""Thin wrapper around djitellopy.Tello for connection + RC + landing."""
import time
from djitellopy import Tello


class Drone:
    def __init__(self, flight_mode: bool, min_battery: int = 20):
        self.flight_mode = flight_mode
        self.min_battery = min_battery
        self.tello = Tello()
        self._last_rc_log = 0.0
        self._log_interval = 0.5

    def connect(self):
        self.tello.connect()
        battery = self.tello.get_battery()
        print(f"[Drone] Battery: {battery}%")
        if battery < self.min_battery:
            raise SystemExit(f"[SAFETY] Battery {battery}% < {self.min_battery}%")
        return battery

    def start_stream(self):
        self.tello.streamon()
        print("[Drone] Video stream ON")

    def takeoff(self):
        print("[Drone] Takeoff...")
        self.tello.takeoff()
        print("[Drone] Takeoff successful")

    def frame_reader(self):
        return self.tello.get_frame_read()

    def hover_kick(self, throttle=50, up_time=1.0, pause=0.5):
        print("[Drone] Initial hover kick ...")
        self.tello.send_rc_control(0, 0, throttle, 0)
        time.sleep(up_time)
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(pause)

    def send_rc(self, lr, fb, ud, yaw, log_interval=0.5):
        now = time.time()
        if now - self._last_rc_log >= log_interval:
            print(f"RC -> LR:{lr:+d} FB:{fb:+d} UD:{ud:+d} YAW:{yaw:+d}")
            self._last_rc_log = now
        if self.flight_mode:
            self.tello.send_rc_control(int(lr), int(fb), int(ud), int(yaw))

    def stop(self):
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

    def land(self):
        if not self.flight_mode:
            return
        try:
            self.tello.land()
            print("[Drone] Landed")
        except Exception as e:
            print(f"[Drone] Land error: {e}")

    def shutdown(self):
        try:
            self.tello.streamoff()
        except Exception:
            pass