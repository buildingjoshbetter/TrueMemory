"""Tests for ThrottlerStateMachine — all with mock data, no GPU needed."""

import time

from truememory.tier_switch.state_machine import ThrottlerStateMachine


def _ok_readings():
    return {
        "mps_level": {"status": "ok", "used_gb": 3.0, "ratio": 0.25},
        "growth_rate": {"status": "ok", "slope_pct": 1.0},
        "thermal": {"status": "ok", "scheduler_limit": 100},
    }


def _warning_mps():
    r = _ok_readings()
    r["mps_level"] = {"status": "warning", "used_gb": 10.5, "ratio": 0.875}
    return r


def _warning_growth():
    r = _ok_readings()
    r["growth_rate"] = {"status": "warning", "slope_pct": 6.0}
    return r


def _critical_thermal():
    r = _ok_readings()
    r["thermal"] = {"status": "critical", "scheduler_limit": 60}
    return r


def _critical_mps():
    r = _ok_readings()
    r["mps_level"] = {"status": "critical", "used_gb": 11.5, "ratio": 0.96}
    return r


def test_starts_at_batch_1():
    sm = ThrottlerStateMachine(start_batch=1, max_batch=12, ramp_step=2)
    assert sm.batch_size == 1
    assert sm.state == ThrottlerStateMachine.PROBING


def test_safety_check_all_ok():
    sm = ThrottlerStateMachine(start_batch=4, max_batch=12, ramp_step=2)
    result = sm.safety_check(_ok_readings())
    assert result == 4
    assert sm.good_streak == 1


def test_safety_check_warning_steps_down():
    sm = ThrottlerStateMachine(start_batch=6, max_batch=12, ramp_step=2)
    result = sm.safety_check(_warning_mps())
    assert result == 4


def test_safety_check_critical_halves():
    sm = ThrottlerStateMachine(start_batch=8, max_batch=12, ramp_step=2)
    result = sm.safety_check(_critical_thermal())
    assert result == 4


def test_step_down_enters_stable():
    sm = ThrottlerStateMachine(start_batch=6, max_batch=12, ramp_step=2)
    sm.safety_check(_warning_mps())
    assert sm.state == ThrottlerStateMachine.STABLE


def test_backoff_enters_backoff():
    sm = ThrottlerStateMachine(start_batch=8, max_batch=12, ramp_step=2)
    sm.safety_check(_critical_mps())
    assert sm.state == ThrottlerStateMachine.BACKOFF


def test_backoff_cooldown_returns_to_probing():
    sm = ThrottlerStateMachine(start_batch=8, max_batch=12, ramp_step=2)
    sm.safety_check(_critical_mps())
    assert sm.state == ThrottlerStateMachine.BACKOFF
    sm.last_backoff_time = time.time() - 121
    sm.good_streak = 3
    assert sm.should_ramp_check() is True
    assert sm.state == ThrottlerStateMachine.PROBING


def test_ramp_requires_3_good_streaks():
    sm = ThrottlerStateMachine(start_batch=4, max_batch=12, ramp_step=2)
    sm.last_ramp_time = time.time() - 200
    sm.good_streak = 2
    assert sm.should_ramp_check() is False
    sm.good_streak = 3
    assert sm.should_ramp_check() is True


def test_ramp_requires_120s_cooldown():
    sm = ThrottlerStateMachine(start_batch=4, max_batch=12, ramp_step=2)
    sm.good_streak = 5
    sm.last_ramp_time = time.time() - 60
    assert sm.should_ramp_check() is False
    sm.last_ramp_time = time.time() - 121
    assert sm.should_ramp_check() is True


def test_ramp_requires_all_channels_ok():
    sm = ThrottlerStateMachine(start_batch=4, max_batch=12, ramp_step=2)
    result = sm.ramp_up(_warning_mps())
    assert result == 4  # no ramp


def test_ramp_increases_by_step():
    sm = ThrottlerStateMachine(start_batch=4, max_batch=12, ramp_step=2)
    result = sm.ramp_up(_ok_readings())
    assert result == 6


def test_ramp_caps_at_max():
    sm = ThrottlerStateMachine(start_batch=11, max_batch=12, ramp_step=2)
    result = sm.ramp_up(_ok_readings())
    assert result == 12


def test_warning_on_growth_rate():
    sm = ThrottlerStateMachine(start_batch=6, max_batch=12, ramp_step=2)
    result = sm.safety_check(_warning_growth())
    assert result == 4
    assert sm.state == ThrottlerStateMachine.STABLE


def test_critical_on_mps_level():
    sm = ThrottlerStateMachine(start_batch=8, max_batch=12, ramp_step=2)
    result = sm.safety_check(_critical_mps())
    assert result == 4
    assert sm.state == ThrottlerStateMachine.BACKOFF


def test_batch_never_below_1():
    sm = ThrottlerStateMachine(start_batch=1, max_batch=12, ramp_step=2)
    result = sm.safety_check(_warning_mps())
    assert result == 1

    sm2 = ThrottlerStateMachine(start_batch=1, max_batch=12, ramp_step=2)
    result = sm2.safety_check(_critical_mps())
    assert result == 1
