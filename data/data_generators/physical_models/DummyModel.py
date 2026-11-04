"""
DummyModel v6
Goal: Minimal three-box simulation for district heating + substation + house.

---------------------------------------------------------------------------
UNITS AND CONVENTIONS (defined here once)
---------------------------------------------------------------------------
Time
- simulation_length_steps: number of discrete simulation steps
- time_step_seconds: seconds per simulation step (default 3600 s = 1 hour)

Temperatures (all in °C)
- primary_supply_temp: district heating supply temp entering the substation
- primary_return_temp: district heating return temp leaving the substation (main efficiency target)
- indoor_air_temp: indoor air temp (state variable)
- outdoor_air_temp: outdoor/ambient temp (scenario driver)
- secondary_return_temp: return temp from building loop back to heat exchanger

Substation and hydraulics
- valve_opening: control action in [0, 1] (dimensionless)
- heat_exchanger_transfer_area: heat exchanger effective transfer area (m^2)
- max_primary_mass_flow: upper bound of primary mass flow (kg/s) when valve_opening = 1

Building physics
- building_thermal_resistance: effective thermal resistance (K/W)
  -> heat_loss_W = (indoor_air_temp - outdoor_air_temp) / building_thermal_resistance

- building_thermal_capacity: effective thermal capacity (J/K)
  -> dT = (time_step_seconds / building_thermal_capacity) * (heat_input_W - heat_loss_W)

Heat exchanger proxy (simplified)
- effective_UA = heat_exchanger_U_base * heat_exchanger_transfer_area   (W/K)
- heat_input_W is limited by:
  (a) exchanger UA limit:      exchanger_UA_limit_W = effective_UA * deltaT
  (b) flow transport limit:    exchanger_flow_limit_W = m_dot * water_specific_heat * deltaT
  where deltaT = max(0, primary_supply_temp - secondary_return_temp)

Scenario config (NOT a per-house static parameter)
- indoor_temp_setpoint: desired indoor temp (°C)
  This can change by user behavior or scenario definition, so we keep it outside self.params.

---------------------------------------------------------------------------
THREE-BOX STRUCTURE
---------------------------------------------------------------------------
Box 1: District Heating Side (Source)
- primary_supply_temp (input)
- primary_return_temp (output, updated each step)

Box 2: Substation (Bridge + Control)
- valve_opening (control action)
- heat_exchanger_transfer_area (static)
- max_primary_mass_flow (static bound)

Box 3: House (Demand + Environment)
- indoor_air_temp (state)
- outdoor_air_temp (scenario driver)
- building_thermal_resistance (static)
- building_thermal_capacity (static)
- secondary_return_temp (state proxy)

Notes:
- All per-house static quantities are stored in self.params and can be set via set_params().
- Setpoint is stored in self.scenario and can be changed without changing the "house identity".
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd


class DummyModel:
    # Parameter bounds in the same spirit as YAML ranges: [lower, upper]
    PARAMETER_RANGES: Dict[str, List[float]] = {
        "primary_supply_temp": [60.0, 120.0],             # °C
        "heat_exchanger_transfer_area": [0.5, 50.0],       # m^2
        "max_primary_mass_flow": [0.01, 20.0],                # kg/s
        "building_thermal_resistance": [1e-15, 0.02],       # K/W
        "building_thermal_capacity": [1e6, 1e10],          # J/K
    }

    # Names of per-house static parameters (house identity)
    ALLOWED_PARAMS = {
        "primary_supply_temp",
        "heat_exchanger_transfer_area",
        "max_primary_mass_flow",
        "building_thermal_resistance",
        "building_thermal_capacity",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}

        # Simulation configuration (not a house parameter)
        self.simulation_length_steps: int = int(self.config.get("simulation_length_steps", 24))
        self.time_step_seconds: float = float(self.config.get("time_step_seconds", 3600.0))  # s

        # Scenario configuration (outdoor_air_temp is here, not in params)
        self.scenario: Dict[str, Any] = {
            "indoor_temp_setpoint": float(self.config.get("indoor_temp_setpoint", 21.0)),  # °C
            "outdoor_air_temp": float(self.config.get("outdoor_air_temp", 5.0)),
        }

        # Per-house static parameters: stored in ONE place (self.params)
        self.params: Dict[str, float] = {}
        self._init_box1_district_params()
        self._init_box2_substation_params()
        self._init_box3_house_params()

        # Internal model constants (tuners for proxy physics)
        self.water_specific_heat: float = float(self.config.get("water_specific_heat", 4180.0))  # J/(kg*K)
        self.heat_exchanger_U_base: float = float(self.config.get("heat_exchanger_U_base", 200.0))  # W/(K*m^2)
        self.secondary_return_smoothing: float = float(self.config.get("secondary_return_smoothing", 0.2))  # [0,1]

        # Initial conditions (not a parameter, just initial state values)
        self.initial_indoor_air_temp: float = float(self.config.get("initial_indoor_air_temp", 10.0))  # °C
        self.initial_primary_return_temp: float = float(self.config.get("initial_primary_return_temp", 60.0))  # °C
        self.initial_secondary_return_temp: float = float(
            self.config.get("initial_secondary_return_temp", self.initial_indoor_air_temp)
        )  # °C
        self.initial_valve_opening: float = float(self.config.get("initial_valve_opening", 0.0))  # [0,1]

        # Start at initial state
        self.reset()

    # ------------------------------------------------------------------
    # Box initializers (store static values into self.params)
    # ------------------------------------------------------------------

    def _init_box1_district_params(self) -> None:
        """Box 1 (District Heating Side): defines the source temperature level."""
        # Björn: We can play with this later in meta-training. Just keep that in mind.
        self.params["primary_supply_temp"] = float(self.config.get("primary_supply_temp", 80.0))  # °C

    def _init_box2_substation_params(self) -> None:
        """Box 2 (Substation): defines the bridge hardware and hydraulic limit."""
        self.params["heat_exchanger_transfer_area"] = float(self.config.get("heat_exchanger_transfer_area", 2.0))  # m^2
        self.params["max_primary_mass_flow"] = float(self.config.get("max_primary_mass_flow", 0.05))  # kg/s

    def _init_box3_house_params(self) -> None:
        """Box 3 (House): defines building envelope and environment driver."""
        self.params["building_thermal_resistance"] = float(self.config.get("building_thermal_resistance", 0.005))  # K/W
        self.params["building_thermal_capacity"] = float(self.config.get("building_thermal_capacity", 8e6))  # J/K

    # ------------------------------------------------------------------
    # Scenario setters (not part of params)
    # ------------------------------------------------------------------

    def set_scenario(self, scenario_updates: Dict[str, Any]) -> None:
        """Update scenario configuration (e.g., setpoint changes, outdoor temp)."""
        if not isinstance(scenario_updates, dict):
            raise TypeError("scenario_updates must be a dict.")
        self.scenario.update(scenario_updates)

    # ------------------------------------------------------------------
    # Control (placeholder policy)
    # ------------------------------------------------------------------

    def _temp_error(self) -> float:
        """Difference between desired indoor temp and current indoor temp."""
        return float(self.scenario["indoor_temp_setpoint"]) - float(self.indoor_air_temp)

    def _rule_based_valve_controller(self, temp_error: float) -> float:
        """
        Simple rule-based controller for valve_opening.
        Demonstrates the closed loop. Replace later with learned policy.
        """
        if temp_error > 0.5:
            return 1.0
        if temp_error > 0.0:
            return 0.3
        return 0.0

    # ------------------------------------------------------------------
    # House loop proxies
    # ------------------------------------------------------------------

    def _update_secondary_return_temp(self) -> float:
        """
        Proxy for the building loop:
        secondary_return_temp moves toward indoor_air_temp.
        """
        k = float(np.clip(self.secondary_return_smoothing, 0.0, 1.0))
        return float(self.secondary_return_temp + k * (self.indoor_air_temp - self.secondary_return_temp))

    def _building_heat_loss_W(self) -> float:
        """
        Heat loss from the building to ambient (W).
        Uses building_thermal_resistance (K/W).
        """
        resistance = max(float(self.params["building_thermal_resistance"]), 1e-12)
        outdoor_temp = float(self.scenario["outdoor_air_temp"])
        return float((self.indoor_air_temp - outdoor_temp) / resistance)

    # ------------------------------------------------------------------
    # Heat exchanger proxy
    # ------------------------------------------------------------------

    def _heat_input_to_house_W(self, next_secondary_return_temp: float) -> float:
        """
        Heat delivered from primary side to secondary side (W), simplified.

        Two limits:
        - UA limit:   effective_UA * deltaT
        - Flow limit: m_dot * water_specific_heat * deltaT
        """
        valve_opening_clipped = float(np.clip(self.valve_opening, 0.0, 1.0))
        primary_mass_flow = float(self.params["max_primary_mass_flow"]) * valve_opening_clipped  # kg/s

        effective_UA = float(self.heat_exchanger_U_base) * max(float(self.params["heat_exchanger_transfer_area"]), 1e-12)
        deltaT = max(0.0, float(self.params["primary_supply_temp"]) - float(next_secondary_return_temp))  # K-equivalent

        exchanger_UA_limit_W = effective_UA * deltaT
        exchanger_flow_limit_W = primary_mass_flow * float(self.water_specific_heat) * deltaT

        return float(min(exchanger_UA_limit_W, exchanger_flow_limit_W))

    def _update_primary_return_temp(self, heat_input_W: float) -> float:
        """
        Primary return temp based on extracted heat (simplified).
        """
        valve_opening_clipped = float(np.clip(self.valve_opening, 0.0, 1.0))
        primary_mass_flow = float(self.params["max_primary_mass_flow"]) * valve_opening_clipped
        if primary_mass_flow <= 1e-12:
            return float(self.primary_return_temp)

        deltaT_primary = float(heat_input_W) / (primary_mass_flow * float(self.water_specific_heat))
        return float(self.params["primary_supply_temp"] - deltaT_primary)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        """
        Run one simulation step:
        1) compute temp_error
        2) choose valve_opening via controller
        3) update secondary_return_temp proxy
        4) compute heat_input_W via exchanger proxy
        5) compute heat_loss_W via building envelope
        6) update indoor_air_temp using energy balance
        7) update primary_return_temp based on extracted heat
        """
        temp_error = self._temp_error()
        self.valve_opening = float(np.clip(self._rule_based_valve_controller(temp_error), 0.0, 1.0))

        next_secondary_return_temp = self._update_secondary_return_temp()
        heat_input_W = self._heat_input_to_house_W(next_secondary_return_temp)
        heat_loss_W = self._building_heat_loss_W()

        capacity = max(float(self.params["building_thermal_capacity"]), 1e-12)
        delta_indoor_temp = (float(self.time_step_seconds) / capacity) * (heat_input_W - heat_loss_W)
        next_indoor_air_temp = float(self.indoor_air_temp + delta_indoor_temp)

        next_primary_return_temp = self._update_primary_return_temp(heat_input_W)

        # Commit updates
        self.secondary_return_temp = next_secondary_return_temp
        self.indoor_air_temp = next_indoor_air_temp
        self.primary_return_temp = next_primary_return_temp

        # Logged signals
        return {
            # District
            "primary_supply_temp": float(self.params["primary_supply_temp"]),
            "primary_return_temp": float(self.primary_return_temp),

            # Substation
            "valve_opening": float(self.valve_opening),
            "heat_exchanger_transfer_area": float(self.params["heat_exchanger_transfer_area"]),
            "max_primary_mass_flow": float(self.params["max_primary_mass_flow"]),

            # House + environment
            "indoor_air_temp": float(self.indoor_air_temp),
            "outdoor_air_temp": float(self.scenario["outdoor_air_temp"]),
            "secondary_return_temp": float(self.secondary_return_temp),
            "building_thermal_resistance": float(self.params["building_thermal_resistance"]),
            "building_thermal_capacity": float(self.params["building_thermal_capacity"]),

            # Scenario
            "indoor_temp_setpoint": float(self.scenario["indoor_temp_setpoint"]),

            # Diagnostics
            "heat_input_W": float(heat_input_W),
            "heat_loss_W": float(heat_loss_W),
            "temp_error": float(temp_error),
        }

    def generate_data(self) -> pd.DataFrame:
        """
        Run the simulation for simulation_length_steps and return a DataFrame.
        """
        rows = []
        for step_index in range(self.simulation_length_steps):
            row = self.step()
            row["time_step_index"] = step_index
            row["time_seconds"] = float(step_index) * float(self.time_step_seconds)
            row["group_id"] = "Group 1"
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def set_params(self, new_params: Dict[str, float], validate_ranges: bool = True) -> None:
        """
        Set per-house static parameters (defines a different house).

        - Only names in ALLOWED_PARAMS can be set.
        - If validate_ranges is True, each value is checked against PARAMETER_RANGES.

        Changing per-house params resets the state trajectory.
        """
        if not isinstance(new_params, dict):
            raise TypeError("new_params must be a dict.")

        for name, value in new_params.items():
            if name not in self.ALLOWED_PARAMS:
                raise KeyError(f"Parameter '{name}' is not allowed. Allowed: {sorted(self.ALLOWED_PARAMS)}")

            val = float(value)

            if validate_ranges and name in self.PARAMETER_RANGES:
                low, high = self.PARAMETER_RANGES[name]
                if not (low <= val <= high):
                    print(low, val, high)
                    raise ValueError(f"Parameter '{name}'={val} out of range [{low}, {high}].")

            self.params[name] = val

        self.reset()

    def get_params(self) -> Dict[str, float]:
        """Return current per-house static parameters."""
        return dict(self.params)

    def reset(self) -> None:
        """
        Reset dynamic states to initial conditions.
        This enables re-running a trajectory with the same house parameters.
        """
        self.indoor_air_temp = float(self.initial_indoor_air_temp)
        self.secondary_return_temp = float(self.initial_secondary_return_temp)
        self.primary_return_temp = float(self.initial_primary_return_temp)
        self.valve_opening = float(np.clip(self.initial_valve_opening, 0.0, 1.0))
