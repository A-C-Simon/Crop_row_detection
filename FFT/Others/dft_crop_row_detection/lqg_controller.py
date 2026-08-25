"""LQG navigation controller from Section 3.2 of the paper.

The paper uses a discrete car-like path-tracking error model (eq. 26-30):

    x_{n+1} = A x_n + B u_n + w,   y_n = C x_n + v

    x = (e_y, e_y', e_theta, e_theta')^T
    u = curvature kappa
    y = (e_y_hat, e_theta_hat)^T  (from the DFT row detector)

    A = [[1, dt, 0, 0],
         [0,  0, s, 0],
         [0,  0, 1, dt],
         [0,  0, 0,  0]]
    B = (0, 0, 0, s)^T

where s is the forward speed and dt the control period. The paper prints a
greek nu in the last entry of B; physically it is the curvature-to-heading-rate
gain equal to the forward speed, so we use s (see kimi.md).

The LQG controller is an LQR state-feedback gain u = -K x_hat applied to the
Kalman-filter estimate x_hat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Discrete LQR (infinite horizon) via the discrete algebraic Riccati equation
# ---------------------------------------------------------------------------

def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
         R: np.ndarray) -> np.ndarray:
    """Discrete-time LQR gain K such that u = -K x minimizes the quadratic cost.

    Solves the DARE  P = A'PA - A'PB (R + B'PB)^-1 B'PA + Q  by fixed-point
    iteration, then K = (R + B'PB)^-1 B'PA.
    """
    P = Q.copy()
    for _ in range(500):
        P_new = A.T @ P @ A - (A.T @ P @ B) @ np.linalg.solve(R + B.T @ P @ B,
                                                              B.T @ P @ A) + Q
        if np.max(np.abs(P_new - P)) < 1e-12:
            P = P_new
            break
        P = P_new
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Linear discrete Kalman filter for x_{n+1} = A x + B u + w, y = C x + v."""

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 Q: np.ndarray, R: np.ndarray,
                 x0: Optional[np.ndarray] = None,
                 P0: Optional[np.ndarray] = None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.zeros(self.A.shape[0]) if x0 is None else np.asarray(x0, float)
        self.P = np.eye(self.A.shape[0]) if P0 is None else np.asarray(P0, float)

    def predict(self, u: np.ndarray) -> None:
        self.x = self.A @ self.x + self.B @ np.asarray(u, float).ravel()
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y: np.ndarray) -> None:
        y = np.asarray(y, float).ravel()
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.solve(S, np.eye(self.R.shape[0]))
        self.x = self.x + K @ (y - self.C @ self.x)
        self.P = (np.eye(self.A.shape[0]) - K @ self.C) @ self.P

    def step(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.predict(u)
        self.update(y)
        return self.x.copy()


# ---------------------------------------------------------------------------
# Controller + closed-loop simulation
# ---------------------------------------------------------------------------

@dataclass
class LQGController:
    """LQR gain + Kalman filter for crop-row tracking.

    Parameters
    ----------
    speed : float          forward speed s (m/s)
    dt : float             control period (s)
    Q_x : array-like       state cost (4x4)
    R_u : float            control cost (curvature)
    Q_w : array-like       process noise covariance (4x4)
    R_v : array-like       measurement noise covariance (2x2)
    """

    speed: float
    dt: float
    Q_x: np.ndarray
    R_u: float
    Q_w: np.ndarray
    R_v: np.ndarray

    def __post_init__(self):
        self.A = np.array([[1.0, self.dt, 0.0, 0.0],
                           [0.0, 0.0, self.speed, 0.0],
                           [0.0, 0.0, 1.0, self.dt],
                           [0.0, 0.0, 0.0, 0.0]])
        self.B = np.array([[0.0], [0.0], [0.0], [self.speed]])
        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])
        self.K = dlqr(self.A, self.B,
                      np.asarray(self.Q_x, float), np.array([[float(self.R_u)]]))
        self.kf = None

    def reset(self, x0: Optional[np.ndarray] = None,
              P0: Optional[np.ndarray] = None) -> None:
        self.kf = KalmanFilter(self.A, self.B, self.C,
                               np.asarray(self.Q_w, float),
                               np.asarray(self.R_v, float),
                               x0=x0, P0=P0)

    def control(self, x_hat: np.ndarray) -> float:
        """u = -K x_hat -> curvature command."""
        return float(-self.K @ np.asarray(x_hat, float).ravel())

    def simulate(self, steps: int, x0: np.ndarray,
                 seed: Optional[int] = None) -> dict:
        """Closed-loop simulation with measurement noise.

        The true plant uses the same model with process noise w ~ N(0, Q_w) and
        measurements y = C x + v, v ~ N(0, R_v). Returns the state history.
        """
        rng = np.random.default_rng(seed)
        self.reset(x0)
        x = np.asarray(x0, float).copy()
        history = {"x": [x.copy()], "x_hat": [self.kf.x.copy()], "u": [0.0]}
        for _ in range(steps):
            u = self.control(self.kf.x)
            # true plant step
            x = self.A @ x + self.B @ np.array([u]) + \
                rng.multivariate_normal(np.zeros(4), self.Q_w)
            y = self.C @ x + rng.multivariate_normal(np.zeros(2), self.R_v)
            self.kf.step([u], y)
            history["x"].append(x.copy())
            history["x_hat"].append(self.kf.x.copy())
            history["u"].append(u)
        for k, v in history.items():
            history[k] = np.array(v)
        return history