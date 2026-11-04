"""
A physical model of the heating system

Created: 2024-10-10
@author: mwiemer


Abbreviations:
bs = buffer storage
gb = gas boiler
hc = heat circuit
hp = heat pump
is = inside (building)
os = outside (building)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple
from numbers import Number
from collections.abc import  Iterable
import  inspect
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.integrate import odeint

logger = logging.getLogger(__name__)

@dataclass
class Params:
    '''
    The collection of physical model parameters.
    Time-dependent parameters are realized as functions.
    '''
    os_temp: Callable = field(default_factory=lambda: (lambda t: None))  # the measured outside temperature
    is_temp_base: Callable = field(default_factory=lambda: (lambda t: None))  # the measured inside temperature
    is_temp_target_delta: Callable = field(default_factory=lambda: (lambda t: None))  # the change in target inside temperature
    is_capacity: float = None  # inside (building) capacity
    is_os_exchange_coeff: float = None  # coefficient linking inside-outside temperature difference to heat power flow from the inside to the outside

    # heating circuitfis_os_exchange_coeff
    hc_pow_base: Callable = field(default_factory=lambda: (lambda t: None))  # measured thermal power of the  heating circuit
    hc_supply_temp: Callable = field(default_factory=lambda: (lambda t: None))  # measured supply temperature
    hc_return_temp: Callable = field(default_factory=lambda: (lambda t: None))  # measured return temperature
    hc_flux_max: float = None  # maximum flux

    # gas boiler
    gb_th_nom_pow_th: float = None  # nominal power of the gas boiler)
    gb_heating_curve: Callable = field(default_factory=lambda: (lambda t: None))  # the target inlet temperature as a function of the outside temperature
    gb_cop: float = None  # ratio of thermal output to gas input for the gas boiler

    # heat pump
    hp_th_pow_1: Callable = field(default_factory=lambda: (lambda t: None))  # output power of the heat pump at step 1
    hp_th_pow_2: Callable = field(default_factory=lambda: (lambda t: None))  # output power of the heat pump at step 2
    hp_cop: Callable = field(default_factory=lambda: (lambda t: None))  # the ratio of thermal output power to electrical input power as a functi0on of the supply temperature


    # buffer storage
    bs_capacity: float = None  # heat capacity *i.e. fraction of energy increase per temperautre increse) of the buffer storage
    bs_top_to_bottom_temp_exchange_coeff: float = None  # coefficient linking top-bottom temperature difference to the energy flow from bottom to top of the buffer storage
    bs_target_temp: Callable = field(default_factory=lambda: (lambda t: None))


class ArrayIndexClass:
    """
    Creates a mapping for attribues of the class to array indices.
    """
    def __init__(self):
        # associate labels with indices and create a list of labels in right order
        index = 0
        self._labels = []
        for k, v in inspect.getmembers(self, lambda obj: not callable(obj)):
            try:
                self._validate_label(k)
            except Exception as _:
                continue
            assert v is None
            setattr(self, k, index)
            self._labels.append(k)
            index += 1
        # set dimension
        self._dimension: int = len(self._labels)

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self):
        return self._dimension

    @property
    def labels(self) -> List[str]:
        return self._labels
    
    def as_array(self, d: Dict[int or str, Number] = {}) -> np.ndarray:
        a = np.zeros(self.dimension, dtype=np.float64)
        for k, v in d.items():
            if isinstance(k, str):
                self._validate_label(k)
                setattr(self, k, v)
            else:
                self._validate_index(k)
                a[k] = v
        return a
    
    def as_dict(self, a: np.ndarray = None, labeled: bool = True) -> Dict[int or str, Number]:
        if a is None:
            a = np.zeros(self.dimension, dtype=np.float64)
        assert len(a) == self.dimension
        return {l: v for l, v in zip(self.labels, a)} if labeled else {i: v for i, v in enumerate(a)}

            
    def _validate_index(self, i) -> None:
        assert isinstance(i, int)
        assert 0 <= i
        assert i < self.dimension
    
    def _validate_label(self, label: str) -> None:
        assert isinstance(label, str)
        assert not label.startswith('_')
        assert label not in ['dimension', 'labels']
        assert  hasattr(self, label)

class CoreStateIndexClass(ArrayIndexClass):
    """ The indices of the mutually independent observables of the heating system """
    time = None  # the current time
    is_temp_delta = None  #difference "simulated" - "measured" inside temperature
    bs_top_temp = None  # SOC at the top half of the buffer storage
    bs_bottom_temp = None  # SOC at the bottom half of the buffer storage
    hp_el_meter = None  # electricity meter for the heat pump
    gb_gas_meter = None  # gas meter for the gas boiler
    hc_heat_meter = None  # the heat meter for the heating circuit  # removed to enhance computational performance

CSIndex = CoreStateIndexClass()

class ActionIndexClass(ArrayIndexClass):
    time = None
    #gb_op_state = None  # continuous variable that controls the heat generation from gas boilers (value < 0 = 'off', value > 1 = 'full load')
    hp_op_state = None  # continuous variable that controls the heat generation from the heat pump (value < 0 = 'off', 1 < value < 2 = 'step 1 of heat pump', value > 2 = 'step 2 of heat pump'
    bs_op_state = None  # continuous variable that controls  heat power extracted from the buffer storage

ActIndex = ActionIndexClass()

class FullState:
    """
    Representation of the full state of the heating system, including the temperatures, power flows, etc.
    All quantities are derived from the core state of the heating system using the model parameters and applied actions.

    The class allows to overwrite the time temporarily which is  useful for integration of the ordinary differential equations.
    The class also allows to load multiple core states and actions in order to assess the full state for multiple time steps in parallel.
    """

    def __init__(self, core_state: np.ndarray, params: Params, actions: np.ndarray = None) -> "FullState":
        self._core_state: np.ndarray = None
        self._actions: np.ndarray = None
        self._params: Params = None
        self._t_overwritten: float = None  # to simulate the same state with a diffeent time (required for the ODE integration)
        self.set_core_state(core_state)
        self.set_actions(np.zeros(ActIndex.dimension) if actions is None else actions)
        self.set_params(params)

    @property
    def core_state(self) -> np.ndarray:
        return self._core_state

    @core_state.setter
    def core_state(self, core_state: np.ndarray):
        shape = core_state.shape
        if len(shape)==2 and shape[1]==CSIndex.dimension:
            core_state = core_state.transpose()
        assert len(core_state) == CSIndex.dimension
        self._core_state = core_state

    def set_core_state(self, core_state: np.ndarray) -> 'FullState':
        self.core_state = core_state
        return self

    @property
    def actions(self) -> np.ndarray:
        return self._actions

    @actions.setter
    def actions(self, actions: np.ndarray):
        if actions is not None:
            shape = actions.shape
            if len(shape) == 2 and shape[1] == ActIndex.dimension:
                actions = actions.transpose()
            assert len(actions) == ActIndex.dimension
        self._actions = actions

    def set_actions(self, actions: np.ndarray) -> 'FullState':
        self.actions = actions
        return self

    @property
    def params(self) -> Params:
        return self._params

    def set_params(self, params: Params) -> 'FullState':
        assert isinstance(params, Params)
        self._params = params
        return self

    def overwrite_time(self, t: float) -> 'FullState':
        """
        To set  the time for evaluation of internal time dependent parameters to to a time different from the true staate time.
        This feature is required for the ODE integration.
        """
        self._t_overwritten = t
        return self

    def remove_overwritten_time(self) -> 'FullState':
        """
        To reset the time for evaluation of internal time dependent parameters to the true state time.
        """
        self._t_overwritten = None
        return self

    @property
    def time(self) -> float:
        return self._core_state[CSIndex.time] if self._t_overwritten is None else self._t_overwritten

    @property
    def os_temp(self) -> float:
        return self._params.os_temp(self.time)

    @property
    def is_temp(self) -> float:
        return self.params.is_temp_base(self.time) + self.is_temp_delta

    @property
    def is_temp_target_delta(self) -> float:
        return self.params.is_temp_target_delta(self.time)

    @property
    def is_temp_delta(self) -> float:
        return self._core_state[CSIndex.is_temp_delta]

    @property
    def is_delta_soc(self) -> float:
        return self.is_temp_delta / self.params.is_capacity

    @property
    def is_temp_base(self) -> float:
        return self.params.is_temp_base(self.time)

    @property
    def is_target_temp(self) -> float:
        return self.is_temp_base + self.is_temp_target_delta

    @property
    def is_to_os_power_thermal_delta(self) -> float:
        return self._params.is_os_exchange_coeff * self.is_temp_delta

    @property
    def is_to_os_power_thermal_target_delta(self) -> float:
        return self._params.is_os_exchange_coeff * self.is_temp_target_delta

    @property
    def is_to_os_power_thermal(self) -> float:
        return self._params.is_os_exchange_coeff * (self.is_temp - self.os_temp)

    @property
    def hc_supply_temp(self) -> float:
        return self.gb_supply_temp

    @property
    def hc_max_supply_temp(self) -> float:
        return np.maximum(self.gb_target_supply_temp, self.bs_top_temp)

    @property
    def hc_return_temp(self) -> float:
        return np.minimum(self.params.hc_return_temp(self.time),
                          self.hc_max_supply_temp - self.hc_power_thermal_base / self.params.hc_flux_max) # to enable that at least the base thermal demand can be covered

    @property
    def hc_supply_return_temp_delta(self) -> float:
        return self.hc_supply_temp - self.hc_return_temp

    @property
    def hc_flux(self) -> float:
        return np.minimum(self.params.hc_flux_max, self.hc_power_thermal_demand / (self.hc_max_supply_temp - self.hc_return_temp))

    @property
    def hc_power_thermal(self) -> float:
        return np.minimum(self.hc_power_thermal_potential, self.hc_power_thermal_demand)

    @property
    def hc_power_thermal_base(self) -> float:
        return self.params.hc_pow_base(self.time)

    @property
    def hc_power_thermal_demand(self) -> float:
        return np.maximum(0, self.hc_power_thermal_base + self.hc_power_thermal_delta_demand)

    @property
    def hc_power_thermal_potential(self) -> float:
        return self.gb_power_thermal_potential + self.bs_power_thermal_potential

    @property
    def hc_power_thermal_delta(self):
        return self.hc_power_thermal - self.hc_power_thermal_base

    @property
    def hc_power_thermal_delta_demand(self) -> float:
        """ the desired additional power (on top of the base power) assuming an infinitly powerful gas boiler """
        # temperature difference to target
        delta = self.is_temp_target_delta - self.is_temp_delta
        # linear regulation for room temperatures below the target temperature
        p1 = ((self.hc_max_supply_temp - self.hc_return_temp)
              * self.params.hc_flux_max * smooth_from_minus1_to_1(delta))
        # remaining power to enable thermal equilibium (zero above target temperature):
        p2 = self.is_to_os_power_thermal_target_delta * smooth_from_0_to_1(5 * delta + 1)
        # combination of p1 and p2:
        return p1 + p2

    @property
    def hc_heat_meter(self) -> float:
        return self.core_state[CSIndex.hc_heat_meter] if hasattr(CSIndex, 'hc_heat_meter') else np.nan

    @property
    def gb_supply_temp(self) -> float:
        return self.gb_return_temp + self.gb_supply_return_temp_delta

    @property
    def gb_target_supply_temp(self) -> float:
        return self.params.gb_heating_curve(self.os_temp)

    @property
    def gb_return_temp(self) -> float:
        return self.bs_supply_temp

    @property
    def gb_supply_return_temp_delta(self) -> float:
        return np.clip(self.gb_target_supply_temp - self.gb_return_temp,
                       0,
                       self.params.gb_th_nom_pow_th/self.params.hc_flux_max)

    @property
    def gb_power_thermal(self) -> float:
        # compensates for what could not be delivered by the heat pump
        return self.hc_flux * self.gb_supply_return_temp_delta

    @property
    def gb_power_thermal_potential(self) -> float:
        return np.clip(
            a=self.params.hc_flux_max * (self.gb_target_supply_temp - self.bs_supply_temp),
            a_min=0,
            a_max=self.params.gb_th_nom_pow_th
        )

    @property
    def gb_power_gas(self) -> float:
        return self.gb_power_thermal / self.params.gb_cop

    @property
    def gb_gas_meter(self) -> float:
        return self.core_state[CSIndex.gb_gas_meter]

    @property
    def hp_power_thermal(self) -> float:
        s = self._actions[ActIndex.hp_op_state]
        p1 = self._params.hp_th_pow_1(self.time)
        p2 = self._params.hp_th_pow_2(self.time)
        return p1 * (s >= 1) + (p2-p1) * (s >= 2)

    @property
    def hp_power_electrical(self) -> float:
        return self.hp_power_thermal / self.params.hp_cop(self.bs_top_temp)

    @property
    def hp_electricity_meter(self) -> None:
        return self.core_state[CSIndex.hp_el_meter]

    @property
    def bs_target_temp(self) -> float:
        return self.params.bs_target_temp(self.time)

    @property
    def bs_supply_temp(self) -> float:
        return self.bs_top_temp

    @property
    def bs_return_temp(self) -> float:
        return self.hc_return_temp

    @property
    def bs_supply_return_temp_delta(self) -> float:
        return self.bs_supply_temp - self.bs_return_temp

    @property
    def bs_power_thermal(self) -> float:
        return self.bs_bottom_to_hc_heat_flow + self.bs_top_to_hc_heat_flow

    @property
    def bs_power_thermal_potential(self) -> float:
        return self.bs_supply_return_temp_delta * self.params.hc_flux_max

    @property
    def bs_soc_thermal(self) -> float:
        return self.bs_top_soc_thermal + self.bs_bottom_soc_thermal

    @property
    def bs_top_soc_thermal(self) -> float:
        return 0.5 * self.params.bs_capacity * (self.bs_top_temp - self.is_temp)

    @property
    def bs_bottom_soc_thermal(self) -> float:
        return 0.5 * self.params.bs_capacity * (self.bs_bottom_temp - self.is_temp)

    @property
    def bs_top_temp(self) -> float:
        return self.core_state[CSIndex.bs_top_temp]

    @property
    def bs_bottom_temp(self) -> float:
        return self.core_state[CSIndex.bs_bottom_temp]

    @property
    def bs_bottom_to_hc_heat_flow(self) -> float:
        return self.hc_flux * (self.bs_bottom_temp - self.hc_return_temp)

    @property
    def bs_top_to_hc_heat_flow(self) -> float:
        return self.hc_flux * (self.bs_top_temp - self.bs_bottom_temp)

    @property
    def bs_top_to_bottom_temp_exchange(self) -> float:
        return self._params.bs_top_to_bottom_temp_exchange_coeff * (self.bs_top_temp - self.bs_bottom_temp)

    @property
    def bs_top_to_bottom_heat_flow(self) -> float:
        return 0.5 * self.params.bs_capacity * self.bs_top_to_bottom_temp_exchange

class PhysicalModel:
    def __init__(self,
                 full_state: FullState,
                 step_time_delta: float = 1,
                 actions_per_step: int = 8):
        self._full_state: FullState = full_state
        self._full_state_tmp: FullState = FullState(
            core_state=np.zeros(CSIndex.dimension),
            params=full_state.params
        )
        assert step_time_delta>0
        self._step_time_delta: float = step_time_delta
        assert isinstance(actions_per_step, int) and actions_per_step > 0
        self._actions_per_step: int = actions_per_step
        self._core_state_log: np.ndarray = None  # chronology of core states
        self._core_state_log_index: int = None  # index of the next entry in the core_state log
        self._actions_log: np.ndarray = None  # chronology of actions
        self._actions_log_index: int = None  # index of the next entry in the actions log
        self.reset()

    def reset(self, core_state: np.ndarray = None) -> 'PhysicalModel':
        """ Resets the physyical model. In particular deletes the history of recent model states. """
        if core_state is not None:
            self.full_state.core_state = core_state
        return self._reset_log()

    @property
    def full_state(self) -> FullState:
        return self._full_state
    
    @property
    def step_time_delta(self) -> float:
        return self._step_time_delta

    @property
    def actions_time_delta(self) -> float:
        return self._step_time_delta / self._actions_per_step

    @property
    def core_state_log(self) -> np.ndarray:
        return None if self._core_state_log is None else self._core_state_log[:self._core_state_log_index]

    @property
    def core_state_log_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self.core_state_log, columns=CSIndex.labels)
        df.set_index(CSIndex.labels[CSIndex.time], inplace=True)
        df.sort_index(inplace=True)
        return df

    @property
    def actions_log(self) -> np.ndarray:
        return None if self._actions_log is None else self._actions_log[:self._actions_log_index]

    @property
    def actions_log_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self.actions_log, columns=ActIndex.labels)
        df.set_index(ActIndex.labels[ActIndex.time], inplace=True)
        df.sort_index(inplace=True)
        return df

    @property
    def full_state_log(self) -> FullState:
        return FullState(
            core_state=self.core_state_log,
            params=self.full_state.params,
            actions=self.actions_log
        )

    def make_step(self) -> 'PhysicalModel':
        """ Lets the system progress by step_time_delta """
        # go ahead step by step updating the actions
        for n in range(self._actions_per_step):
            # get the rule-based action for the initial state:
            self._update_actions()
            # log the initial state
            self._log()
            # integrate to the next time step for a rule-based action
            with warnings.catch_warnings(record=True) as catched_warnings:
                t = [self.full_state.time, self._full_state.time + self.actions_time_delta]
                odeint_output = odeint(
                    func=self._calculate_core_state_derivative,
                    y0=_remove_t_from_core_state(self.full_state.core_state),
                    t=t,
                    full_output=True,
                    atol=1e-2, # default is  1.49012e-8
                    rtol=1e-2,  # default is  1.49012e-8
                    h0=5/3600  # coresponds to 5 seconds
                )

                new_core_state = _add_t_to_core_state(odeint_output[0][-1], t[-1])
                if catched_warnings:
                    message = (
                        'Raised warning     : ' + str(catched_warnings) +
                        '\nCore state before: ' + str(CSIndex.as_dict(self.full_state.core_state)) +
                        '\nCore state now   : ' + str(CSIndex.as_dict(new_core_state)) +
                        '\nodeint full outpu: ' + str(odeint_output[1])
                    )
                    logger.warn(message)
                self.full_state.core_state = new_core_state
        return self

    def get_core_state_log_index(self, t: float) -> int:
        return int(np.round((t - self._core_state_log[0, CSIndex.time]) / self.actions_time_delta))

    def _update_actions(self) -> 'PhysicalModel':
        state = self._full_state
        p = state._params

        # get the full state for the last 10 minutes:
        c_states_last_10_min = self._get_latest(array=self.core_state_log, time_delta=1/6)
        actions_last_10_min = self._get_latest(array=self.actions_log, time_delta=1/6)
        states_last_10_min = (None if (c_states_last_10_min is None or actions_last_10_min is None)
                       else self._full_state_tmp)
        if states_last_10_min is not None:
            states_last_10_min.core_state = c_states_last_10_min
            states_last_10_min.actions = actions_last_10_min

        # updaste time stamp
        state.actions[ActIndex.time] = state.time

        # control heat pump
        # note: the heat pump is controlled based on the measured temperatures of the topand bottom layer of the buffer storage for the last 10 minutes
        if states_last_10_min is None:
            # do nothing
            pass
        else:
            hp_op_state = state._actions[ActIndex.hp_op_state]
            if hp_op_state==0:
                # turn on 1st stage if bs_top_temp below bs_target_temp for more than 1 minute
                if np.all(
                        self._get_latest(array=states_last_10_min.bs_top_temp, time_delta=1/60)
                        < state.bs_target_temp - 2):
                    state.actions[ActIndex.hp_op_state] = 1
            elif hp_op_state==1:
                # turn on 2nd stage if bs_top_temp below bs_target_temp - 2 for more than 10 minutes
                # turn off 1st stage if bs_bottom_temp > bs_target_temp + 2 for more than 1 minute
                if np.all(states_last_10_min.bs_top_temp < state.bs_target_temp - 2):
                    state.actions[ActIndex.hp_op_state] = 2
                # rule to regulate down:
                elif np.all(
                    self._get_latest(states_last_10_min.bs_bottom_temp, time_delta=1/60)
                    >= state.bs_target_temp + 2
                ):
                    state._actions[ActIndex.hp_op_state] = 0
            elif hp_op_state==2:
                # turn off 2nd stage if bs_bottom_temp > bs_target_temp - 5 for more than 10 minutes
                if np.all(states_last_10_min.bs_bottom_temp > state.bs_target_temp - 5):
                    state._actions[ActIndex.hp_op_state] = 1

        return self

    def _to_index_delta(self, time_delta) -> int:
        """ translates a time delta into an index delta assuming a time resolution of step_time_delta/actions_per_step """
        r = time_delta/self.actions_time_delta
        return int(np.round(r))

    def _get_latest(self, array: np.ndarray, time_delta: float) -> np.ndarray:
        ''' returns the latests elements assuming a time resolution of step_time_delta/actions_per_step'''
        index_delta = self._to_index_delta(time_delta)
        return None if array is None else array[-min(index_delta, len(array)):]


    def _calculate_core_state_derivative(self, core_state_without_t: np.ndarray = None, t: float = None) -> np.ndarray:
        # make shortcuts to frequently used objects
        state = self.full_state
        p = state._params
        if core_state_without_t is not None:
            core_state_copy = state.core_state.copy()
            state.core_state = _add_t_to_core_state(core_state_without_t, core_state_copy[CSIndex.time])
        if t is not None:
            state.overwrite_time(t)
        actions = self.full_state.actions

        # instantiate  state derivative
        ds_dt = np.zeros(CSIndex.dimension)

        # update time (not necessary any more)
        # ds_dt[CSIndex.time] = 1

        # change in difference of inside temperature from base value:
        ds_dt[CSIndex.is_temp_delta] = (state.hc_power_thermal_delta
                                        - state.is_to_os_power_thermal_delta) / p.is_capacity

        # change in temperature at the top half of the buffer storage
        top_to_bottom_pow = state.bs_top_to_bottom_heat_flow
        ds_dt[CSIndex.bs_top_temp] = (state.hp_power_thermal - state.bs_top_to_hc_heat_flow - top_to_bottom_pow) / (0.5 * p.bs_capacity)

        # change in temperature at the bottom half of the buffer storage
        ds_dt[CSIndex.bs_bottom_temp] = (top_to_bottom_pow - state.bs_bottom_to_hc_heat_flow) / (0.5 * p.bs_capacity)

        # electricity consumption of heat pump
        ds_dt[CSIndex.hp_el_meter] = state.hp_power_electrical

        # fuel consumption of gas boiler
        ds_dt[CSIndex.gb_gas_meter] = state.gb_power_gas

        # generated heat
        if hasattr(CSIndex, 'hc_heat_meter'):
            ds_dt[CSIndex.hc_heat_meter] = state.hc_power_thermal

        if core_state_without_t is not None:
            state.core_state = core_state_copy
        if t is not None:
            state.remove_overwritten_time()
        return _remove_t_from_core_state(ds_dt)



    def _log_core_state(self) -> 'PhysicalModel':
        if self._core_state_log is None:
            self._core_state_log = np.zeros((1000, CSIndex.dimension))
            self._core_state_log_index = 0
        elif self._core_state_log_index == len(self._core_state_log):
            # extend array
            new = np.zeros((len(self._core_state_log) + 1000, CSIndex.dimension))
            new[:self._core_state_log_index] = self._core_state_log
            del self._core_state_log
            self._core_state_log = new
        self._core_state_log[self._core_state_log_index] = self.full_state.core_state
        self._core_state_log_index += 1
        return self

    def _reset_core_state_log(self) -> 'PhysicalModel':
        self._core_state_log = None
        self._core_state_log_index = None
        return self

    def _log_actions(self) -> 'PhysicalModel':
        if self._actions_log is None:
            self._actions_log = np.zeros((1000, ActIndex.dimension))
            self._actions_log_index = 0
        elif self._actions_log_index == len(self._actions_log):
            # extend array
            new = np.zeros((len(self._actions_log) + 1000, ActIndex.dimension))
            new[:self._actions_log_index] = self._actions_log
            del self._actions_log
            self._actions_log = new
        self._actions_log[self._actions_log_index] = self.full_state.actions
        self._actions_log_index += 1
        return self

    def _reset_actions_log(self) -> 'PhysicalModel':
        self._actions_log = None
        self._actions_log_index = None
        return self

    def _log(self) -> 'PhysicalModel':
        """ Writes the current model state into the log file such that it can be evaluated for the model history """
        return self._log_core_state()._log_actions()

    def _reset_log(self) -> 'PhysicalModel':
        return self._reset_core_state_log()._reset_actions_log()


def smooth_from_0_to_1(x):
    """
    Smooth function to clip x to the value range [0, 1]
    The error is below 1%.
    """
    return  (np.tanh(5 * (x - 0.5)) + 1) / 2

def smooth_from_minus1_to_1(x):
    """
    Smooth function to clip x to the value range [-1, 1]
    The error is below 1%.
    """
    return  np.tanh(3 * x)

if CSIndex.time+1==CSIndex.dimension:
    _add_t_to_core_state = np.append
    _remove_t_from_core_state = lambda core_state: core_state[:-1]
else:
    _add_t_to_core_state = lambda core_state_without_t, t: np.insert(core_state_without_t, t, CSIndex.time)
    _remove_t_from_core_state = lambda core_state: np.delete(core_state, CSIndex.time)


