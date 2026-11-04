"""
DummyModel v5
Goal: Minimal three box simulation using exactly 10 core parameters.

Box 1 (District): T_p_s, T_p_r
Box 2 (Substation): alpha, A
Box 3 (House): T_int, T_amb, T_set, R_build, C_build, T_s_r

No task_id.

Notes:
- R_build is interpreted as K/W so that Q_loss = (T_int - T_amb) / R_build is in Watts.
- alpha is clipped to [0, 1].
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class DummyModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}
        self.length: int = int(self.config.get("length", 24))
        # Todo: Name is not at all self-explaining.
        self.dt_s: float = float(self.config.get("dt_s", 3600.0))

        # Set the ten core parameters
        
        # Todo Vinod: Put all parameters that change for different houses but are static for one particular houses
        #   into a dict called self.params = {}. This is important for set and get functions at the botom of this class.
        self.params = {} # Empty dict for changable parameters determining a specific house.
        self.init_Box1()
        self.init_Box2()
        self.init_Box3()

        # Minimal internal tuners (constants)
        self.cp: float = float(self.config.get("cp", 4180.0))            # J/(kg*K)
        self.mass_flow_rate: float = float(self.config.get("mass_flow_rate", 0.05))            # kg/s when alpha = 1
        self.U_base: float = float(self.config.get("U_base", 200.0))     # W/(m2*K)
        self.k_r: float = float(self.config.get("k_r", 0.2))             # per-step smoothing [0,1]

        # @Vinod: Look into reset function at the very botom.
        #   These values are no parameters as they configure the class rather than specifying building properties.
        self.initial_inside_temp = config.get('initial_inside_temp', 15)
        self.initial_outside_temp = config.get('initial_outside_temp', 5)
        # Todo: Yet, this variable is not needed.
        self.time: np.ndarray = np.arange(self.length, dtype=float)

        # Start a new "epoch".
        self.reset()

    def init_Box1(self):
        ''' Initializes Box1, i.e., districted heating network side.
        '''
        # Todo: Change naming to easier understandable names or list names in fucntion description if you need them from a heating point of view.
        self.T_p_s: float = float(self.config.get("T_p_s", 80.0))
        self.T_p_r: float = float(self.config.get("T_p_r_init", 60.0))

    def init_Box2(self):
        ''' Initializes Box2, i.e., house substation.
        '''
        self.A: float = float(self.config.get("A", 2.0))
        self.alpha: float = float(self.config.get("alpha_init", 0.0))

    def init_Box3(self):
        ''' Initializes Box 3, i.e. House (state, environment, static building parameters)
        '''
        self.T_amb: float = float(self.config.get("T_amb", 5.0))
        self.T_set: float = float(self.config.get("T_set", 21.0))
        self.R_build: float = float(self.config.get("R_build", 0.8))     # K/W
        self.C_build: float = float(self.config.get("C_build", 8e6))     # J/K
        self.T_s_r: float = float(self.config.get("T_s_r_init", self.T_int))
        

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def compute_error(self) -> float:
        return self.T_set - self.T_int

    # ------------------------------------------------------------------
    # House loop proxies
    # ------------------------------------------------------------------

    def update_secondary_return(self) -> float:
        """
        Proxy: secondary return moves toward indoor temperature.
        k_r is a per-step smoothing factor in [0, 1].
        """
        k = float(np.clip(self.k_r, 0.0, 1.0))
        return self.T_s_r + k * (self.T_int - self.T_s_r)

    def house_loss(self) -> float:
        """
        Building heat loss Q_loss in Watts.
        R_build is interpreted as K/W.
        """
        return (self.T_int - self.T_amb) / max(self.R_build, 1e-12)

    # ------------------------------------------------------------------
    # Heat exchanger proxy
    # ------------------------------------------------------------------

    def heat_transfer(self, T_s_r_next: float) -> float:
        """
        Heat exchanger transfer Q_in in Watts.
        Limited by exchanger UA and by transported heat via mass flow.
        """
        alpha_clipped = float(np.clip(self.alpha, 0.0, 1.0))
        m_dot_p = self.k_m * alpha_clipped

        UA = self.U_base * max(self.A, 1e-9)
        dT_hex = max(0.0, self.T_p_s - T_s_r_next)

        Q_hex_UA = UA * dT_hex
        Q_hex_flow = m_dot_p * self.cp * dT_hex

        return min(Q_hex_UA, Q_hex_flow)

    def update_primary_return(self, Q_in: float) -> float:
        """
        Primary return temperature update based on extracted heat.
        """
        alpha_clipped = float(np.clip(self.alpha, 0.0, 1.0))
        m_dot_p = self.k_m * alpha_clipped
        if m_dot_p <= 1e-12:
            return self.T_p_r

        dT_p = Q_in / (m_dot_p * self.cp)
        return self.T_p_s - dT_p

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, float]:
        """
        One simulation step.
        Returns a dict of signals for logging.
        """
        error = self.compute_error()
        self.alpha = controller_alpha(error)

        T_s_r_next = self.update_secondary_return()
        Q_in = self.heat_transfer(T_s_r_next)
        Q_loss = self.house_loss()

        dT_int = (self.dt_s / max(self.C_build, 1e-12)) * (Q_in - Q_loss)
        T_int_next = self.T_int + dT_int # in words: dt_int quals delta_inside_temp

        T_p_r_next = self.update_primary_return(Q_in)

        # Commit updates
        self.T_s_r = T_s_r_next
        self.T_int = T_int_next
        self.T_p_r = T_p_r_next

        return {
            # 10 core parameters (logged each step)
            "T_p_s": self.T_p_s,
            "T_p_r": self.T_p_r,
            "alpha": float(np.clip(self.alpha, 0.0, 1.0)),
            "A": self.A,
            "T_int": self.T_int,
            "T_amb": self.T_amb,
            "T_set": self.T_set,
            "R_build": self.R_build,
            "C_build": self.C_build,
            "T_s_r": self.T_s_r,
            # extra helpful diagnostics
            "Q_in_W": Q_in,
            "Q_loss_W": Q_loss,
            "error": error,
        }

    def generate_data(self) -> pd.DataFrame:
        """
        Runs the simulation for self.length steps and returns a DataFrame.
        """
        rows = []
        for k in range(self.length):
            out = self.step()
            out["t_idx"] = k # Todo: Does this set the time variable?
            rows.append(out)
        return pd.DataFrame(rows)

    def set_params(self, params):
        ''' Assigns the given parameters to the respective variables inside this class.
        Note: Changing parameters through this function creates a (theoretically) new building.

        Args: params: A dict with values for the different parameters.
        '''
        # Todo: We need to write a list of allowed parameters so that one cannot set nonsense names with some dict from outside this class.
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)
        self.reset()

    def reset(self):
        ''' Resets the class to initial time point
        so that one can generate new data for the same time period.
        '''
        self.inside_temp = self.initial_inside_temp
        # Note: Outside temp is constant in this version. This is included nevertheless.
        self.outside_temp = self.initial_outside_temp


    def get_params(self):
        ''' Returns the current parameters as a dict.
        '''
        return self.params


def controller_alpha(self, error: float) -> float:
    """
    Helper function:
    Simple rule-based controller for alpha.
    This is only to demonstrate the closed loop.
    Later replace with learned policy.
    """
    if error > 0.5:
        return 1.0
    if error > 0.0:
        return 0.3
    return 0.0
