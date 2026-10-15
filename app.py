"""Main application class — wires everything together."""
import time
import os
import cv2
import numpy as np
import torch
import keyboard

from .config import AppConfig
from .detector import Detector
from .aligner import Aligner
from .hud import HUD
from .drone import Drone


class WindowPassApp:
    def __init__(self, cfg: AppConfig = None):
        self.cfg = cfg or AppConfig()
        self._setup_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = Detector(
            model_path=self.cfg.model.path,
            confidence=self.cfg.model.confidence,
            inference_size=self.cfg.model.inference_size,
            max_detections=self.cfg.model.max_detections,
            device=self.device,
            min_mask_area=self.cfg.runtime.min_mask_area,
        )
        self.aligner = Aligner(self.cfg)
        self.hud = HUD(self.cfg.camera)
        self.drone = Drone(self.cfg.runtime.flight_mode,
                           self.cfg.runtime.min_battery)

        self._fps_last_time = time.perf_counter()
        self._fps_frames = 0
        self._fps = 0.0
        self._last_status_log = 0.0

    # ------------------------------------------------------------------
    def _setup_torch(self):
        if not torch.cuda.is_available():
            try:
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        cv2.setNumThreads(2)

    # ------------------------------------------------------------------
    def run(self):
        try:
            self.drone.connect()
            self.drone.start_stream()
            self.drone.takeoff()
            frame_read = self.drone.frame_reader()
            self.drone.hover_kick()
            print("[App] Starting main loop...")
            self._loop(frame_read)
        except KeyboardInterrupt:
            print("\n[App] Interrupted")
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    def _loop(self, frame_read):
        while True:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.001)
                continue

            frame = cv2.resize(
                frame,
                (self.cfg.camera.frame_width, self.cfg.camera.frame_height),
                interpolation=cv2.INTER_AREA,
            )

            camera_center = self.hud.camera_center(frame.shape)

            # Detection
            detection = self.detector.detect(frame)
            self.detector.annotate(frame, detection)

            # Aligner
            info = self.aligner.update(detection, frame.shape, camera_center)

            # FPS
            self._update_fps()

            # HUD
            self.hud.draw(frame, info,
                          window_center=detection.center,
                          camera_center=camera_center,
                          fps=self._fps,
                          inference_size=self.cfg.model.inference_size)

            # Logging
            self._log_status(info)

            # RC
            self.drone.send_rc(info.roll_cmd, info.forward_cmd,
                               info.throttle_cmd, info.yaw_cmd,
                               self.cfg.runtime.log_interval)

            # UI
            cv2.imshow("Window PASS", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[App] Q pressed")
                break
            if key == ord("r"):
                self.aligner.reset()
                print("[App] Reset")
            if keyboard.is_pressed("e"):
                print("[EMERGENCY] E pressed")
                break

    # ------------------------------------------------------------------
    def _update_fps(self):
        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_last_time = now

    def _log_status(self, info):
        now = time.time()
        if now - self._last_status_log < self.cfg.runtime.log_interval:
            return
        print(f"[{info.stage}] MODE:{info.control_mode} "
              f"LR:{info.roll_cmd:+d} FB:{info.forward_cmd:+d} "
              f"UD:{info.throttle_cmd:+d} YAW:{info.yaw_cmd:+d} | "
              f"FPS:{self._fps:.1f}")
        self._last_status_log = now

    # ------------------------------------------------------------------
    def _cleanup(self):
        print("[App] Cleaning up...")
        self.drone.stop()
        self.drone.land()
        self.drone.shutdown()
        cv2.destroyAllWindows()
        print("[App] Program finished")