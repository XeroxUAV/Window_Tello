from ultralytics import YOLO
from djitellopy import Tello
import cv2
import numpy as np
import time
import torch
import keyboard

# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = r"D:\Window_Tello\best.pt"
CONFIDENCE = 0.5
FLIGHT_MODE = True
MIN_BATTERY = 20

SEARCH_YAW_SPEED = 20
SEARCH_360_TIME = 12.0
SEARCH_ALTERNATE = True
SEARCH_UP_AFTER_360 = True
SEARCH_UP_DURATION = 2.0
SEARCH_UP_THROTTLE = 20
MAX_UP_COUNT = 5

MIN_MASK_AREA = 2000

# =====================================================
# YAW + ROLL — همزمان از مرکز پنجره
# =====================================================
# --- YAW ---
YAW_THRESHOLD = 15.0
YAW_STABLE_COUNT = 3

# --- ROLL (نرمالایز از مرکز پنجره) ---
ROLL_THRESHOLD = 0.04
ROLL_EXIT_THRESHOLD = 0.06
ROLL_STABLE_COUNT = 3
ENABLE_ROLL = True
ROLL_INVERT = True               # ✅ اصلاح شد: True (علامت برعکس)

# --- THROTTLE ---
THROTTLE_THRESHOLD = 13.0
THROTTLE_STABLE_COUNT = 3

# =====================================================
# PID SETTINGS — YAW (سریع‌تر)
# =====================================================
YAW_MAX_SPEED = 35
YAW_DEADZONE = 15.0
YAW_KP = 0.03
YAW_KI = 0.00
YAW_KD = 0.0
YAW_INTEGRAL_LIMIT = 100.0

# =====================================================
# PID SETTINGS — ROLL (آروم‌تر)
# =====================================================
ROLL_MAX_SPEED = 18
ROLL_DEADZONE = 0.04
ROLL_KP = 2
ROLL_KI = 0.0
ROLL_KD = 0.0
ROLL_INTEGRAL_LIMIT = 20.0

# =====================================================
# PID SETTINGS — THROTTLE
# =====================================================
THR_KP = 0.4
THR_KI = 0.05
THR_KD = 0.0
THR_INTEGRAL_LIMIT = 100.0
THR_MAX_SPEED = 25

# --- حداقل سرعت‌ها ---
MIN_APPLICABLE_SPEED_THR = 10
MIN_APPLICABLE_SPEED_YAW = 9
MIN_APPLICABLE_SPEED_ROLL = 9

# --- APPROACH ---
APPROACH_SPEED = 20
APPROACH_CENTER_DEADZONE_YAW = 10
APPROACH_CENTER_DEADZONE_THR = 9
APPROACH_CENTER_DEADZONE_ROLL = 0.03
APPROACH_CENTER_MAX = 15

AREA_RATIO_THRESHOLD = 0.35
AREA_INCREASE_THRESHOLD = 1.02
APPROACH_MAX_TIME = 20.0

# --- PASS ---
PASS_FORWARD_SPEED = 40
PASS_EXTRA_TIME = 1.25
PASS_MAX_TIME = 15.0
PASS_AREA_DROP = 0.45

# --- Camera Offset ---
CAMERA_CENTER_OFFSET_X = 0
CAMERA_CENTER_OFFSET_Y = 0


# =====================================================
# PID CONTROLLER
# =====================================================
class PID:
    def __init__(self, Kp=0.08, Ki=0.0001, Kd=0.015,
                 integral_limit=100.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit

        self.prev_error = 0
        self.integral = 0
        self.last_output = 0
        self.last_time = time.time()

    def update(self, error):
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.last_time = now

        self.integral += error * dt
        self.integral = np.clip(self.integral,
                                -self.integral_limit,
                                self.integral_limit)

        derivative = (error - self.prev_error) / dt

        output = (self.Kp * error
                  + self.Ki * self.integral
                  + self.Kd * derivative)

        output = 0.7 * self.last_output + 0.3 * output

        self.prev_error = error
        self.last_output = output

        return output

    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.last_output = 0
        self.last_time = time.time()


# =====================================================
# HELPERS
# =====================================================
def get_camera_center(frame_shape):
    """مرکز دوربین = وسط فریم + افست"""
    h, w = frame_shape[:2]
    cx = w // 2 + CAMERA_CENTER_OFFSET_X
    cy = 118
    return (cx, cy)


def calculate_yaw_error(window_center, camera_center):
    """Yaw = فاصله افقی مرکز پنجره از مرکز دوربین (پیکسل)"""
    if window_center is None or camera_center is None:
        return None
    return float(window_center[0] - camera_center[0])


def calculate_throttle_error(window_center, camera_center):
    """Throttle = فاصله عمودی مرکز پنجره از مرکز دوربین (پیکسل)"""
    if window_center is None or camera_center is None:
        return None
    return float(window_center[1] - camera_center[1])


def calculate_roll_error(window_center, frame_shape):
    """
    Roll = عدم تعادل افقی پنجره نسبت به لبه‌های فریم.

    حالا با فرمول اصلاح‌شده:
      roll_error = (dist_right - dist_left) / total
      + ROLL_INVERT

    نتیجه:
      پنجره راست → roll_err مثبت → roll_cmd مثبت (برو راست)
      پنجره چپ  → roll_err منفی → roll_cmd منفی (برو چپ)
    """
    if window_center is None:
        return None, None, None
    h, w = frame_shape[:2]
    cx = window_center[0]
    dist_left = cx
    dist_right = w - cx
    total = dist_left + dist_right
    if total == 0:
        return 0.0, dist_left, dist_right

    # ✅ فرمول اصلاح‌شده (R - L به‌جای L - R)
    roll_error = float(dist_right - dist_left) / total

    if ROLL_INVERT:
        roll_error = -roll_error

    return roll_error, int(dist_left), int(dist_right)


# =====================================================
# MAIN ALIGNER
# =====================================================
class Aligner:
    def __init__(self):
        # ---- YAW ----
        self.yaw_speed = SEARCH_YAW_SPEED
        self.yaw_direction = 1
        self.yaw_stable_count = 0
        self.yaw_centered = False
        self.EXIT_YAW_THRESHOLD = YAW_THRESHOLD * 2

        self.search_start_time = None
        self.search_direction = 1
        self.up_after_search = False
        self.up_start_time = None
        self.up_count = 0
        self.MAX_UP_COUNT_LOCAL = MAX_UP_COUNT

        self.pid_yaw = PID(Kp=YAW_KP, Ki=YAW_KI, Kd=YAW_KD,
                           integral_limit=YAW_INTEGRAL_LIMIT)

        # ---- ROLL (از مرکز پنجره) ----
        self.roll_stable_count = 0
        self.roll_centered = not ENABLE_ROLL
        self.EXIT_ROLL_THRESHOLD = ROLL_EXIT_THRESHOLD
        self.pid_roll = PID(Kp=ROLL_KP, Ki=ROLL_KI, Kd=ROLL_KD,
                            integral_limit=ROLL_INTEGRAL_LIMIT)

        # ---- THROTTLE ----
        self.throttle_stable_count = 0
        self.throttle_centered = False
        self.EXIT_THROTTLE_THRESHOLD = THROTTLE_THRESHOLD * 2
        self.pid_throttle = PID(Kp=THR_KP, Ki=THR_KI, Kd=THR_KD,
                                integral_limit=THR_INTEGRAL_LIMIT)

        # APPROACH / PASS
        self.approach_start_time = None
        self.max_area_seen = 0
        self.area_prev = 0
        self.pass_start_time = None
        self.pass_initial_area = None
        self.pass_previous_area = None
        self.pass_area_increased = False
        self.pass_extra_time = None

        # smoothing
        self.smoothed_yaw = None
        self.smoothed_throttle = None
        self.smoothed_roll = None
        self.ALPHA = 0.3

        # lost
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = 25
        self.stage = "YAW"

        # نمایش
        self.last_dist_left = None
        self.last_dist_right = None
        self.control_mode = "SIMULTANEOUS"

    def reset(self):
        self.__init__()

    # ---------------------------------------------
    def update(self, window_center, bbox, area, frame_shape, camera_center):

        # =====================================================
        # PASS
        # =====================================================
        if self.stage == "PASS":
            now = time.time()

            if self.pass_start_time is None:
                self.pass_start_time = now
                self.pass_initial_area = area if area is not None else 0
                self.pass_previous_area = self.pass_initial_area
                self.pass_area_increased = False
                self.pass_extra_time = None
                print(f"[PASS] 🚁 START (initial_area={self.pass_initial_area})")

            if self.pass_extra_time is not None:
                elapsed_extra = now - self.pass_extra_time
                if elapsed_extra >= PASS_EXTRA_TIME:
                    print(f"[PASS] ✅ DONE")
                    self.stage = "DONE"
                    return self._make(None, None, None, area,
                                      0, 0, 0, 0,
                                      "DONE", True, True, True, True)
                return self._make(None, None, None, area,
                                  0, 0, PASS_FORWARD_SPEED, 0,
                                  f"PASS extra ({elapsed_extra:.1f}/{PASS_EXTRA_TIME}s)",
                                  True, True, True, False)

            elapsed = now - self.pass_start_time

            if (self.pass_previous_area is not None
                    and self.pass_previous_area > 0
                    and area is not None
                    and area > self.pass_previous_area * 1.05):
                self.pass_area_increased = True

            if (self.pass_area_increased
                    and elapsed > 0.5
                    and (area is None
                         or area < self.pass_initial_area * PASS_AREA_DROP)):
                print(f"[PASS] 🎯 Passing detected")
                self.pass_extra_time = now

            if area is not None:
                self.pass_previous_area = area

            if elapsed >= PASS_MAX_TIME:
                print(f"[PASS] ⚠️ Timeout → DONE")
                self.stage = "DONE"
                return self._make(None, None, None, area,
                                  0, 0, 0, 0,
                                  "DONE (timeout)", True, True, True, True)

            return self._make(None, None, None, area,
                              0, 0, PASS_FORWARD_SPEED, 0,
                              f"PASS ({elapsed:.1f}s, area={area})",
                              True, True, True, False)

        # =====================================================
        # DONE
        # =====================================================
        if self.stage == "DONE":
            return self._make(None, None, None, area,
                              0, 0, 0, 0,
                              "DONE", True, True, True, True)

        # =====================================================
        # پنجره گم
        # =====================================================
        if window_center is None:
            now = time.time()

            if self.up_after_search:
                elapsed_up = now - self.up_start_time
                if elapsed_up >= SEARCH_UP_DURATION:
                    print(f"[SEARCH] ⬆️ Up done → restart 360")
                    self.up_after_search = False
                    self.up_start_time = None
                    self.search_start_time = now
                    self.search_direction = 1
                    self.up_count += 1
                    if self.up_count >= self.MAX_UP_COUNT_LOCAL:
                        print(f"[SEARCH] ⚠️ MAX UP reached")
                        self.MAX_UP_COUNT_LOCAL = 999
                    return self._make(None, None, None, None,
                                      0, 0, 0, 0,
                                      "SEARCH 360 (restart)",
                                      False, False, False, False)
                return self._make(None, None, None, None,
                                  0, SEARCH_UP_THROTTLE, 0, 0,
                                  f"SEARCH UP ({elapsed_up:.1f}/{SEARCH_UP_DURATION}s, #{self.up_count})",
                                  False, False, False, False)

            if self.search_start_time is None:
                self.search_start_time = now
                print(f"[SEARCH] 🔄 Start 360 (dir={self.search_direction})")

            elapsed_search = now - self.search_start_time

            if elapsed_search >= SEARCH_360_TIME:
                print(f"[SEARCH] ✅ 360 completed")
                if SEARCH_UP_AFTER_360:
                    print(f"[SEARCH] ⬆️ Going UP")
                    self.up_after_search = True
                    self.up_start_time = now
                    self.search_start_time = None
                    return self._make(None, None, None, None,
                                      0, SEARCH_UP_THROTTLE, 0, 0,
                                      f"SEARCH UP (start, #{self.up_count + 1})",
                                      False, False, False, False)
                else:
                    if SEARCH_ALTERNATE:
                        self.search_direction *= -1
                    self.search_start_time = now
                    elapsed_search = 0

            yaw_cmd = self.search_direction * self.yaw_speed
            self.yaw_centered = False
            if ENABLE_ROLL:
                self.roll_centered = False
            self.throttle_centered = False
            self.stage = "SEARCH 360"
            self.yaw_stable_count = 0
            self.roll_stable_count = 0
            self.throttle_stable_count = 0
            self.pid_yaw.reset()
            self.pid_roll.reset()
            self.pid_throttle.reset()

            return self._make(None, None, None, None,
                              yaw_cmd, 0, 0, 0,
                              f"SEARCH 360 ({elapsed_search:.1f}/{SEARCH_360_TIME}s)",
                              False, False, False, False)

        # =====================================================
        # پنجره دیده شد
        # =====================================================
        self.lost_frames = 0

        if self.search_start_time is not None:
            print(f"[SEARCH] 🎯 Window found — reset 360")
            self.search_start_time = None

        if self.up_after_search:
            print(f"[SEARCH] 🎯 Window found during UP — cancel")
            self.up_after_search = False
            self.up_start_time = None

        # ---- خطاها (همه از مرکز پنجره) ----
        yaw_err_raw = calculate_yaw_error(window_center, camera_center)
        thr_err_raw = calculate_throttle_error(window_center, camera_center)
        roll_err_raw, dist_left, dist_right = calculate_roll_error(
            window_center, frame_shape
        )
        self.last_dist_left = dist_left
        self.last_dist_right = dist_right

        # ---- smoothing ----
        if self.smoothed_yaw is None:
            self.smoothed_yaw = yaw_err_raw
        else:
            self.smoothed_yaw = (1 - self.ALPHA) * self.smoothed_yaw + self.ALPHA * yaw_err_raw

        if self.smoothed_throttle is None:
            self.smoothed_throttle = thr_err_raw
        else:
            self.smoothed_throttle = (1 - self.ALPHA) * self.smoothed_throttle + self.ALPHA * thr_err_raw

        if self.smoothed_roll is None:
            self.smoothed_roll = roll_err_raw
        else:
            self.smoothed_roll = (1 - self.ALPHA) * self.smoothed_roll + self.ALPHA * roll_err_raw

        yaw_err = self.smoothed_yaw
        thr_err = self.smoothed_throttle
        roll_err = self.smoothed_roll

        yaw_cmd = 0
        throttle_cmd = 0
        forward_cmd = 0
        roll_cmd = 0

        # =====================================================
        # مرحله ۱: YAW + ROLL — کاملاً همزمان
        # =====================================================
        if not (self.yaw_centered and self.roll_centered):
            self.stage = "YAW+ROLL"
            self.control_mode = "SIMULTANEOUS"

            # ---------- YAW (سریع‌تر) ----------
            if not self.yaw_centered:
                if abs(yaw_err) < YAW_THRESHOLD:
                    yaw_cmd = 0
                    self.pid_yaw.reset()
                    self.yaw_stable_count += 1
                    if self.yaw_stable_count >= YAW_STABLE_COUNT:
                        self.yaw_centered = True
                        print(f"[INFO] ✅ YAW CENTERED (err={yaw_err:+.1f}px)")
                else:
                    pid_out = self.pid_yaw.update(yaw_err)
                    yaw_cmd = int(np.clip(pid_out, -YAW_MAX_SPEED, YAW_MAX_SPEED))
                    if abs(yaw_cmd) < MIN_APPLICABLE_SPEED_YAW and abs(yaw_err) > YAW_DEADZONE:
                        yaw_cmd = (MIN_APPLICABLE_SPEED_YAW if yaw_err > 0
                                   else -MIN_APPLICABLE_SPEED_YAW)
                    self.yaw_stable_count = 0
            else:
                if abs(yaw_err) > self.EXIT_YAW_THRESHOLD:
                    print(f"[INFO] 🔄 EXIT YAW (err={yaw_err:+.1f}px)")
                    self.yaw_centered = False
                    self.yaw_stable_count = 0
                    self.pid_yaw.reset()

            # ---------- ROLL (آروم‌تر، همزمان با Yaw) ----------
            if ENABLE_ROLL:
                if not self.roll_centered:
                    if abs(roll_err) < ROLL_THRESHOLD:
                        roll_cmd = 0
                        self.pid_roll.reset()
                        self.roll_stable_count += 1
                        if self.roll_stable_count >= ROLL_STABLE_COUNT:
                            self.roll_centered = True
                            print(f"[INFO] ✅ ROLL CENTERED (err={roll_err:+.3f})")
                    else:
                        # با فرمول اصلاح‌شده: roll_err>0 → پنجره راست → برو راست → lr مثبت
                        pid_out = self.pid_roll.update(roll_err)
                        roll_cmd = int(np.clip(pid_out, -ROLL_MAX_SPEED, ROLL_MAX_SPEED))
                        if abs(roll_cmd) < MIN_APPLICABLE_SPEED_ROLL and abs(roll_err) > ROLL_DEADZONE:
                            roll_cmd = (MIN_APPLICABLE_SPEED_ROLL if roll_err > 0
                                        else -MIN_APPLICABLE_SPEED_ROLL)
                        self.roll_stable_count = 0
                else:
                    if abs(roll_err) > self.EXIT_ROLL_THRESHOLD:
                        print(f"[INFO] 🔄 EXIT ROLL (err={roll_err:+.3f})")
                        self.roll_centered = False
                        self.roll_stable_count = 0
                        self.pid_roll.reset()

            if self.yaw_centered and self.roll_centered:
                print(f"[INFO] 🎯 YAW + ROLL ALIGNED (simultaneous)")

        # =====================================================
        # مرحله ۲: THROTTLE
        # =====================================================
        if (self.yaw_centered and self.roll_centered
                and not self.throttle_centered):
            self.stage = "THROTTLE"

            if abs(thr_err) < THROTTLE_THRESHOLD:
                throttle_cmd = 0
                self.pid_throttle.reset()
                self.throttle_stable_count += 1
                if self.throttle_stable_count >= THROTTLE_STABLE_COUNT:
                    self.throttle_centered = True
                    print(f"[INFO] ✅ THROTTLE CENTERED (err={thr_err:+.1f}px)")
            else:
                pid_out = self.pid_throttle.update(-thr_err)
                throttle_cmd = int(np.clip(pid_out, -THR_MAX_SPEED, THR_MAX_SPEED))
                if abs(throttle_cmd) < MIN_APPLICABLE_SPEED_THR and abs(thr_err) > MIN_APPLICABLE_SPEED_THR:
                    throttle_cmd = (-MIN_APPLICABLE_SPEED_THR if thr_err > 0
                                    else MIN_APPLICABLE_SPEED_THR)
                self.throttle_stable_count = 0

        elif self.throttle_centered:
            if abs(thr_err) > self.EXIT_THROTTLE_THRESHOLD:
                print(f"[INFO] 🔄 EXIT THROTTLE (err={thr_err:+.1f}px)")
                self.throttle_centered = False
                self.throttle_stable_count = 0
                self.pid_throttle.reset()

        # =====================================================
        # مرحله ۳: APPROACH
        # =====================================================
        if (self.yaw_centered and self.roll_centered
                and self.throttle_centered
                and self.stage != "APPROACH"):
            self.stage = "APPROACH"
            self.approach_start_time = time.time()
            self.max_area_seen = area if area is not None else 0
            self.area_prev = self.max_area_seen
            print(f"[INFO] 🎯 START APPROACH (area={self.max_area_seen})")

        if self.stage == "APPROACH":
            now = time.time()
            elapsed = now - self.approach_start_time

            # YAW در APPROACH (سریع‌تر)
            if abs(yaw_err) > APPROACH_CENTER_DEADZONE_YAW:
                yaw_cmd = int(np.clip(
                    self.pid_yaw.update(yaw_err),
                    -APPROACH_CENTER_MAX, APPROACH_CENTER_MAX
                ))
                if abs(yaw_cmd) < MIN_APPLICABLE_SPEED_YAW:
                    yaw_cmd = (MIN_APPLICABLE_SPEED_YAW if yaw_err > 0
                               else -MIN_APPLICABLE_SPEED_YAW)

            # ROLL در APPROACH (آروم‌تر، همزمان)
            if ENABLE_ROLL and abs(roll_err) > APPROACH_CENTER_DEADZONE_ROLL:
                roll_cmd = int(np.clip(
                    self.pid_roll.update(roll_err),
                    -APPROACH_CENTER_MAX, APPROACH_CENTER_MAX
                ))
                if abs(roll_cmd) < MIN_APPLICABLE_SPEED_ROLL:
                    roll_cmd = (MIN_APPLICABLE_SPEED_ROLL if roll_err > 0
                                else -MIN_APPLICABLE_SPEED_ROLL)

            # THROTTLE در APPROACH
            if abs(thr_err) > APPROACH_CENTER_DEADZONE_THR:
                throttle_cmd = int(np.clip(
                    self.pid_throttle.update(-thr_err),
                    -APPROACH_CENTER_MAX, APPROACH_CENTER_MAX
                ))
                if abs(throttle_cmd) < MIN_APPLICABLE_SPEED_THR:
                    throttle_cmd = (-MIN_APPLICABLE_SPEED_THR if throttle_cmd > 0
                                    else MIN_APPLICABLE_SPEED_THR)

            forward_cmd = APPROACH_SPEED

            stop_approach = False
            stop_reason = ""

            if area is None:
                print(f"[INFO] ⚠️ Lost during APPROACH")
                self.stage = "YAW+ROLL"
                self.yaw_centered = False
                if ENABLE_ROLL:
                    self.roll_centered = False
                self.throttle_centered = False
                self.pid_yaw.reset()
                self.pid_roll.reset()
                self.pid_throttle.reset()
                return self._make(yaw_err, roll_err, thr_err, None,
                                  0, 0, 0, 0,
                                  "YAW+ROLL (lost)",
                                  False, False, False, False)

            total_pixels = frame_shape[0] * frame_shape[1]
            area_ratio = area / total_pixels

            if area_ratio >= AREA_RATIO_THRESHOLD:
                stop_approach = True
                stop_reason = f"area_ratio={area_ratio:.3f}"

            self.area_prev = area

            if stop_approach:
                print(f"[INFO] 🛑 STOP APPROACH ({stop_reason})")
                print(f"[INFO] 🚀 STARTING PASS")
                self.stage = "PASS"
                self.pass_start_time = None
                self.pass_initial_area = None
                self.pass_previous_area = None
                self.pass_area_increased = False
                self.pass_extra_time = None
                self.pid_yaw.reset()
                self.pid_roll.reset()
                self.pid_throttle.reset()
                return self._make(yaw_err, roll_err, thr_err, area,
                                  0, 0, PASS_FORWARD_SPEED, 0,
                                  f"PASS START ({stop_reason})",
                                  True, True, True, False)

            return self._make(yaw_err, roll_err, thr_err, area,
                              yaw_cmd, throttle_cmd, forward_cmd, roll_cmd,
                              f"APPROACH ({elapsed:.1f}s, area={area})",
                              True, True, True, False)

        return self._make(yaw_err, roll_err, thr_err, area,
                          yaw_cmd, throttle_cmd, forward_cmd, roll_cmd,
                          self.stage,
                          self.yaw_centered, self.roll_centered,
                          self.throttle_centered, False)

    # ---------------------------------------------
    def _make(self, yaw_err, roll_err, thr_err, area,
              yaw_cmd, throttle_cmd, forward_cmd, roll_cmd,
              stage, yaw_c, roll_c, thr_c, pass_done):
        return {
            "yaw_error": yaw_err,
            "roll_error": roll_err,
            "throttle_error": thr_err,
            "area": area,
            "yaw_cmd": int(yaw_cmd),
            "roll_cmd": int(roll_cmd),
            "throttle_cmd": int(throttle_cmd),
            "forward_cmd": int(forward_cmd),
            "stage": stage,
            "yaw_centered": yaw_c,
            "roll_centered": roll_c,
            "throttle_centered": thr_c,
            "pass_completed": pass_done,
            "control_mode": getattr(self, "control_mode", "IDLE"),
        }


# =====================================================
# DRAW
# =====================================================
def draw_info(frame, info, window_center=None, camera_center=None,
              dist_left=None, dist_right=None):
    h, w = frame.shape[:2]
    if camera_center is None:
        camera_center = get_camera_center(frame.shape)

    ccx, ccy = camera_center

    # مرکز دوربین (قرمز)
    cv2.circle(frame, (ccx, ccy), 8, (0, 0, 255), -1)
    cv2.circle(frame, (ccx, ccy), 14, (0, 0, 255), 2)
    cv2.putText(frame, "CAM", (ccx + 18, ccy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.line(frame, (ccx, 0), (ccx, h), (0, 0, 255), 1)
    cv2.line(frame, (0, ccy), (w, ccy), (0, 0, 255), 1)

    # مرکز پنجره
    if window_center is not None:
        wcx, wcy = window_center
        cv2.circle(frame, (wcx, wcy), 8, (0, 255, 0), -1)
        cv2.circle(frame, (wcx, wcy), 14, (0, 255, 0), 2)
        cv2.putText(frame, "WIN", (wcx + 18, wcy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # خط Yaw (زرد)
        cv2.line(frame, (ccx, ccy), (wcx, wcy), (255, 255, 0), 2)

        if dist_left is not None:
            cv2.putText(frame, f"L={dist_left}px", (10, wcy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        if dist_right is not None:
            cv2.putText(frame, f"R={dist_right}px", (w - 100, wcy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # HUD
    stage = info["stage"]
    stage_colors = {
        "YAW": (0, 200, 255), "YAW+ROLL": (0, 200, 255),
        "THROTTLE": (0, 165, 255), "APPROACH": (255, 150, 0),
        "PASS": (0, 255, 0), "DONE": (0, 255, 0),
        "SEARCH": (255, 0, 255),
    }
    color = stage_colors.get(stage.split()[0], (255, 255, 255))

    cv2.putText(frame, f"STAGE: {stage}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # MODE
    cm = info.get("control_mode", "IDLE")
    cm_color = (0, 255, 255) if cm == "SIMULTANEOUS" else (200, 200, 200)
    cv2.putText(frame, f"MODE: {cm}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cm_color, 2)

    ye = info.get("yaw_error")
    if ye is not None:
        c = (0, 255, 0) if info.get("yaw_centered") else (0, 255, 255)
        cv2.putText(frame,
                    f"YAW = {ye:+.1f}px  CMD = {info.get('yaw_cmd', 0):+d}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    re = info.get("roll_error")
    if re is not None:
        c = (0, 255, 0) if info.get("roll_centered") else (255, 0, 255)
        cv2.putText(frame,
                    f"ROLL = {re:+.3f}  CMD = {info.get('roll_cmd', 0):+d}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    te = info.get("throttle_error")
    if te is not None:
        c = (0, 255, 0) if info.get("throttle_centered") else (0, 255, 255)
        cv2.putText(frame,
                    f"THR = {te:+.1f}px  CMD = {info.get('throttle_cmd', 0):+d}",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    a = info.get("area")
    if a is not None:
        total = h * w
        ratio = a / total
        cv2.putText(frame, f"AREA = {a} ({ratio*100:.1f}%)",
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

    if info.get("pass_completed", False):
        cv2.putText(frame, "DONE!",
                    (w // 2 - 60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


# =====================================================
# TELLO + YOLO
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)

me = Tello()
me.connect()
battery = me.get_battery()
print(f"[INFO] Battery: {battery}%  Device: {DEVICE}")

if battery < MIN_BATTERY:
    print("[SAFETY] Battery low! Cancelled.")
    exit()

me.streamon()
print("[INFO] Video stream ON")
print("[INFO] ✈️ TAKEOFF ENABLED")

me.takeoff()
print("[INFO] ✅ Takeoff successful")

frame_read = me.get_frame_read()
aligner = Aligner()


# =====================================================
# SEND RC
# =====================================================
def send_rc(lr, fb, ud, yaw):
    print(f"RC → LR:{lr:+d} FB:{fb:+d} UD:{ud:+d} YAW:{yaw:+d}")
    if FLIGHT_MODE:
        me.send_rc_control(int(lr), int(fb), int(ud), int(yaw))


# =====================================================
# MAIN LOOP
# =====================================================
print("[INFO] Starting main loop...")
frame_counter = 0

try:
    while True:
        frame = frame_read.frame
        if frame is None:
            continue
        frame = cv2.resize(frame, (640, 480))

        if frame_counter == 5:
            me.send_rc_control(0, 0, 50, 0)
            time.sleep(1)
            me.send_rc_control(0, 0, 0, 0)

        frame_counter += 1

        h, w = frame.shape[:2]
        camera_center = get_camera_center(frame.shape)

        results = model(frame, verbose=False, conf=CONFIDENCE)

        best_box = None
        best_area = 0
        best_conf = 0

        for result in results:
            if result.boxes is None:
                continue
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                if conf < CONFIDENCE:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)
                    best_conf = conf

        window_center = None
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            wx = (x1 + x2) // 2
            wy = (y1 + y2) // 2
            window_center = (wx, wy)

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 200, 255), 2)
            cv2.putText(frame, f"conf={best_conf:.2f} area={best_area}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 255), 1)

        # Roll از مرکز پنجره (نه IMU)
        info = aligner.update(window_center, best_box, best_area,
                              frame.shape, camera_center)

        draw_info(frame, info,
                  window_center=window_center,
                  camera_center=camera_center,
                  dist_left=getattr(aligner, "last_dist_left", None),
                  dist_right=getattr(aligner, "last_dist_right", None))

        print(f"[{info['stage']}] MODE:{info.get('control_mode', 'IDLE')} "
              f"LR:{info.get('roll_cmd', 0):+d} "
              f"FB:{info.get('forward_cmd', 0):+d} "
              f"UD:{info.get('throttle_cmd', 0):+d} "
              f"YAW:{info.get('yaw_cmd', 0):+d}")

        send_rc(
            lr=info.get("roll_cmd", 0),
            fb=info.get("forward_cmd", 0),
            ud=info.get("throttle_cmd", 0),
            yaw=info.get("yaw_cmd", 0),
        )

        cv2.imshow("Window PASS — fast", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Q pressed")
            break
        if key == ord("r"):
            aligner.reset()
            print("[INFO] 🔄 Reset")

        if keyboard.is_pressed("e"):
            print("[EMERGENCY] E pressed")
            break

except KeyboardInterrupt:
    print("\n[INFO] Keyboard interrupt")

finally:
    print("[INFO] Cleaning up...")
    try:
        me.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass

    try:
        if FLIGHT_MODE:
            me.land()
            print("[INFO] ✅ Landed")
    except Exception as e:
        print(f"[ERROR] Land: {e}")

    try:
        me.streamoff()
    except Exception:
        pass

    cv2.destroyAllWindows()
    print("[INFO] Program finished")