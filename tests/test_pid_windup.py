"""Tests of the PID controller anti-windup and time base, without Home Assistant."""

import importlib.util
from pathlib import Path

PID_FILE = (
    Path(__file__).resolve().parent.parent
    / "custom_components" / "smart_thermostat" / "pid_controller" / "__init__.py"
)
_spec = importlib.util.spec_from_file_location("pid_under_test", PID_FILE)
pid_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pid_module)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_pid(clock, kp=10.0, ki=0.01, kd=0.0, ke=0.0, out_min=0.0, out_max=100.0, max_dt=None):
    return pid_module.PID(
        kp, ki, kd, ke, out_min, out_max, 0, 0.3, 0.3, max_dt=max_dt, time_func=clock
    )


def run(pid, clock, temp, setpoint, seconds, step=60, ext=None):
    out = None
    for _ in range(int(seconds // step)):
        clock.advance(step)
        out, _ = pid.calc(temp, setpoint, ext_temp=ext)
    return out


def test_first_calculation_integrates_with_zero_dt():
    clock = Clock()
    pid = make_pid(clock)
    pid.calc(18.0, 21.0)
    assert pid.integration_dt == 0
    assert pid.integral == 0.0


def test_a_long_saturation_does_not_cause_a_large_overshoot():
    clock = Clock()
    pid = make_pid(clock, kp=20.0, ki=0.02, max_dt=120)
    run(pid, clock, 15.0, 21.0, 48 * 3600)  # saturated at 100 % for two days
    assert pid.integral <= 100.0
    # Setpoint reached: output must leave the saturation at once instead of staying at 100 %
    out = run(pid, clock, 21.0, 21.0, 60)
    assert out < 100.0
    # slightly above the setpoint the output drops to zero within minutes
    out = run(pid, clock, 21.3, 21.0, 600)
    assert out <= 0.0


def test_integral_unwinds_with_negative_error_below_zero_output():
    clock = Clock()
    pid = make_pid(clock, kp=50.0, ki=0.05, max_dt=120)
    run(pid, clock, 20.0, 21.0, 1800)
    start = pid.integral
    assert start > 0
    # large overshoot: P and I give a negative total output
    run(pid, clock, 23.0, 21.0, 120)
    assert pid.integral < start
    previous = pid.integral
    for _ in range(10):
        run(pid, clock, 23.0, 21.0, 120)
        assert pid.integral <= previous
        previous = pid.integral
    # the total output stays below zero the whole time, the integral still unwinds to zero, not beyond
    assert pid.integral == 0.0


def test_negative_output_range_allows_integral_to_unwind_both_ways():
    clock = Clock()
    pid = make_pid(clock, kp=20.0, ki=0.05, out_min=-100.0, out_max=0.0, max_dt=120)
    # cooling: negative error (too warm) pushes the output down to the limit
    run(pid, clock, 24.0, 21.0, 3600)
    assert pid.integral < 0
    wound = pid.integral
    # too cold now: the integral must come back up
    run(pid, clock, 20.5, 21.0, 600)
    assert pid.integral > wound


def test_irregular_sensor_updates_make_no_jumps():
    clock = Clock()
    pid = make_pid(clock, kp=10.0, ki=0.01, max_dt=120)
    run(pid, clock, 20.8, 21.0, 600, step=30)
    before = pid.integral
    # sensor silent for 5 hours, then one calculation
    clock.advance(5 * 3600)
    pid.calc(20.8, 21.0)
    assert pid.integration_dt == 120
    assert abs(pid.integral - before) <= 0.01 * 0.2 * 120 + 1e-9
    # updates after a few seconds integrate only the elapsed seconds
    before = pid.integral
    clock.advance(3)
    pid.calc(20.8, 21.0)
    assert abs(pid.integral - before - 0.01 * 0.2 * 3) < 1e-9


def test_sensor_timestamps_do_not_influence_the_integral():
    clock = Clock()
    pid = make_pid(clock, max_dt=600)
    pid.calc(20.0, 21.0, 100.0, 0.0)
    clock.advance(10)
    pid.calc(20.0, 21.0, 5000.0, 100.0)
    assert abs(pid.integral - 0.01 * 1.0 * 10) < 1e-9


def test_ki_zero_keeps_integral_at_zero_with_external_compensation():
    clock = Clock()
    pid = make_pid(clock, kp=10.0, ki=0.0, ke=2.0, max_dt=120)
    for temp in (18.0, 21.0, 24.0):
        out = run(pid, clock, temp, 21.0, 3600, ext=-25.0)
    assert pid.integral == 0.0
    assert out == 10.0 * (21.0 - 24.0)
    assert pid.external == 2.0 * 46.0


def test_integral_is_clamped_with_external_part():
    clock = Clock()
    pid = make_pid(clock, kp=0.0, ki=1.0, ke=1.0, max_dt=120)
    run(pid, clock, 10.0, 21.0, 3600, ext=1.0)  # external = 20
    assert pid.integral <= 100.0 - 20.0 + 1e-9
    pid.integral = 500.0
    clock.advance(60)
    pid.calc(21.0, 21.0, ext_temp=1.0)
    assert pid.integral <= 80.0 + 1e-9


def test_saturation_takes_external_part_into_account():
    """#326: a positive external part must not freeze the integral at a raw output below zero."""
    clock = Clock()
    pid = make_pid(clock, kp=10.0, ki=0.01, ke=1.0, out_min=0.0, out_max=100.0, max_dt=120)
    pid.calc(21.2, 21.0, ext_temp=-9.0)  # external = 30
    clock.advance(60)
    pid.calc(20.0, 21.0, ext_temp=-9.0)
    clock.advance(60)
    pid.calc(21.2, 21.0, ext_temp=-9.0)  # raw P+I+D < 0 but total > 0
    before = pid.integral
    clock.advance(60)
    pid.calc(21.2, 21.0, ext_temp=-9.0)
    assert pid.integral < before


def test_off_mode_does_not_integrate_and_keeps_the_integral():
    clock = Clock()
    pid = make_pid(clock, max_dt=120)
    run(pid, clock, 20.0, 21.0, 600)
    learned = pid.integral
    assert learned > 0
    pid.mode = "OFF"
    run(pid, clock, 20.0, 21.0, 3600)
    assert pid.integral == learned
    pid.mode = "AUTO"
    clock.advance(1)
    pid.calc(20.0, 21.0)
    assert pid.integration_dt == 0
    assert pid.integral == learned


def test_reset_time_base_keeps_integral_clear_samples_resets_it():
    clock = Clock()
    pid = make_pid(clock, max_dt=120)
    run(pid, clock, 20.0, 21.0, 600)
    learned = pid.integral
    pid.reset_time_base()
    clock.advance(500)
    pid.calc(20.0, 21.0)
    assert pid.integration_dt == 0 and pid.integral == learned
    pid.clear_samples()
    assert pid.integral == 0.0


def test_small_ki_is_accepted_and_integrates_slowly():
    clock = Clock()
    pid = make_pid(clock, kp=0.0, ki=1e-5, max_dt=120)
    pid.set_pid_param(ki=1e-6)
    assert pid._Ki == 1e-6
    pid.set_pid_param(ki=1e-5)
    assert pid._Ki == 1e-5
    run(pid, clock, 20.0, 21.0, 3600)  # error 1 K for 1 h
    assert abs(pid.integral - 1e-5 * 3600) < 1e-3
