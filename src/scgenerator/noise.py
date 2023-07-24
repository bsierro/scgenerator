from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def irfftfreq(freq: np.ndarray, retstep: bool = False):
    """
    Given an array of positive only frequency, this returns the corresponding time array centered
    around 0 that will be aligned with the `numpy.fft.irfft` of a spectrum aligned with `freq`.
    if `retstep` is True, the sample spacing is returned as well
    """
    df = freq[1] - freq[0]
    nt = (len(freq) - 1) * 2
    period = 1 / df
    dt = period / nt

    t = np.linspace(-(period - dt) / 2, (period - dt) / 2, nt)
    if retstep:
        return t, dt
    else:
        return t


def log_power(x):
    return 10 * np.log10(np.abs(np.where(x == 0, 1e-7, x)))


def integrated_rin(freq: np.ndarray, psd: np.ndarray) -> float:
    """
    given a normalized spectrum, computes the total rms RIN in the provided frequency window
    """
    return np.sqrt(cumulative_trapezoid(np.abs(psd)[::-1], -freq[::-1], initial=0)[::-1])


@dataclass
class NoiseMeasurement:
    freq: np.ndarray
    psd: np.ndarray
    phase: np.ndarray | None = None
    psd_interp: interp1d = field(init=False)
    is_uniform: bool = field(default=False, init=False)

    def __post_init__(self):
        df = np.diff(self.freq)
        if df.std() / df.mean() < 1e-12:
            self.is_uniform = True
        self.psd_interp = interp1d(
            self.freq, self.psd, fill_value=(0, self.psd[-1]), bounds_error=False
        )

    @classmethod
    def from_dBc(cls, freq: np.ndarray, psd_dBc: np.ndarray) -> NoiseMeasurement:
        psd = 10 ** (psd_dBc / 10)
        return cls(freq, psd)

    @classmethod
    def from_dBm(
        cls, freq: np.ndarray, psd_dBm: np.ndarray, ref: float, impedence: float = 50.0
    ) -> NoiseMeasurement:
        ref_dB = 10 * np.log10(ref**2 / impedence * 1000)
        psd = 10 ** ((psd_dBm - ref_dB) / 10)
        return cls(freq, psd)

    @classmethod
    def from_time_series(cls, time: np.ndarray, signal: np.ndarray) -> NoiseMeasurement:
        freq = np.fft.rfftfreq(len(time), time[1] - time[0])
        dt = time[1] - time[0]
        psd = np.fft.rfft(signal) / np.sqrt(0.5 * len(time) / dt)
        return cls(freq, psd.real**2 + psd.imag**2, phase=np.angle(psd))

    def sample_spectrum(self, nt: int, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        sample an amplitude spectrum with nt points. The corresponding sample spacing in the time
        is 1/freq.max().

        Parameters
        ----------
        nt : int
            number of points to sample
        dt : float | None, optional
            if given, freq will only be sampled up to 0.5/dt. if that value is higher than the
            max of freq, an exception is raised.
        """
        if nt % 2:
            raise ValueError(f"nt must be an even number, got {nt!r}")

        fmax = 0.5 / dt if dt is not None else self.freq.max()
        if fmax > self.freq.max():
            raise ValueError(
                f"{dt=} yields a max frequency of {fmax:g}Hz, but data only"
                f" goes up to {self.freq.max():g}Hz"
            )

        f = np.linspace(0, fmax, nt // 2 + 1)
        return f, self.psd_interp(f)

    def time_series(
        self, nt: int | np.ndarray, phase: np.ndarray | None = None, dt: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        computes a pulse train whose psd matches the measurement

        Parameters
        ----------
        nt_or_phase : int
            number of points to sample. must be even.
        phase : np.ndarray, shape (nt//2)+1 | None, optional
            phase to apply to the amplitude spectrum before taking the inverse Fourier transform.
            if None, A random phase is then applied.
        dt : float | None, optional
            if given, choose a sample spacing of dt instead of 1/f_max
        """

        freq, spec = self.sample_spectrum(nt, dt)
        if phase is None:
            phase = 2 * np.pi * np.random.rand(len(freq))
        time, dt = irfftfreq(freq, True)

        amp = np.sqrt(spec) * np.exp(1j * phase)
        signal = np.fft.irfft(amp) * np.sqrt(0.5 * len(time) / dt)

        return time, signal

    def integrated_rin(self) -> np.ndarray:
        """
        returns the integrated RIN as fuction of frequency.
        The 0th component is the total RIN in the frequency range covered by the measurement
        """
        return integrated_rin(self.freq, self.psd)
