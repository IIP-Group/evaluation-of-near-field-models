import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

from src.constants import FREE_WAVE_IMPEDANCE, SPEED_OF_LIGHT
from src.hfss import RisRadiationStructure, hfss_ris_parameters

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


def find_rectangle_dim(x_z_idxes: np.ndarray) -> tuple[int, int]:
    jj = x_z_idxes[:, 1][0]
    tol = 1e-7
    dim2 = ((jj - tol < x_z_idxes[:, 1]) & (x_z_idxes[:, 1] < jj + tol)).sum()
    dim1 = len(x_z_idxes) // dim2

    return dim1, dim2


def read_touchstone_file(file: str, freq: float) -> rf.Network:
    rf_net = rf.Network(file=file)
    idx = (np.abs(rf_net.f - freq)).argmin()
    if np.abs(rf_net.f[idx] - freq) > 0.1:
        raise Exception(f"frequency {freq} not file")

    return rf_net[idx : idx + 1]


class FloquetPort(object):
    def __init__(self, freq: float) -> None:
        self.reflection_coeficients, self.capacitances = self.init_floquet_port(
            freq=freq
        )

    def init_floquet_port(self, freq: float) -> tuple[np.ndarray, np.ndarray]:
        net = read_touchstone_file(
            os.path.join(ROOT_PATH, "data", "ris_floquet", "ris_floquet.s3p"), freq=freq
        )

        diode_r = 1.2
        diode_l = 0.4e-9
        r = []

        cap = np.linspace(0.15e-13 * 1e12, 2.1e-11 * 1e12, 1001)

        for c in cap:
            impedance = 1j * 2 * np.pi * freq * diode_l + diode_r / (
                1j * 2 * np.pi * freq * c / 1e12
            )

            s_t = ((impedance) - 50) / (impedance + 50)

            l1 = rf.Network(s=s_t, z0=50, name="l1", frequency=net.frequency)
            l2 = rf.Network(s=s_t, z0=50, name="l2", frequency=net.frequency)

            p1 = rf.Circuit.Port(
                z0=FREE_WAVE_IMPEDANCE, name="p1", frequency=net.frequency
            )

            cnx = [
                [(net, 0), (l1, 0)],
                [(net, 1), (l2, 0)],
                [(p1, 0), (net, 2)],
            ]

            cir = rf.Circuit(cnx)
            reduced_net = cir.network
            r.append(reduced_net.s[0, 0, 0])

        return np.array(r), cap

    def get_capacitance(self, phase_angle: float) -> float:
        phase_angle = phase_angle % (2 * np.pi)
        response = np.angle(self.reflection_coeficients) % (2 * np.pi)
        idx = np.abs(response - phase_angle).argmin()
        return self.capacitances[idx]

    def get_reflection_coeficient(self, cap: float) -> float:
        idx = np.abs(self.capacitances - cap).argmin()
        return self.reflection_coeficients[idx]


class SphericalWaveModel(object):
    def __init__(self, floquet_freq: float) -> None:
        self.floque_port = FloquetPort(freq=floquet_freq)

        self.distance = 0.03

        self.num_antennas_x = 32
        self.num_antennas_y = 4

        self.position_x = (
            self.distance
            * np.linspace(
                -self.num_antennas_x / 2 + 1 / 2,
                self.num_antennas_x / 2 - 1 / 2,
                self.num_antennas_x,
            )[::-1]
        )
        self.position_y = self.distance * np.linspace(
            -self.num_antennas_y / 2 + 1 / 2,
            self.num_antennas_y / 2 - 1 / 2,
            self.num_antennas_y,
        )

    def compute_spherical_wave_capacitances(
        self, tx_theta: float, tx_phi: float, x_z_target: np.ndarray, freq: float
    ) -> np.ndarray:
        tx_theta = np.deg2rad(tx_theta)
        tx_phi = np.deg2rad(tx_phi)
        x, z = x_z_target[0], x_z_target[1]

        wavelength = SPEED_OF_LIGHT / freq
        wavenumber = 2 * np.pi / wavelength

        caps = []
        idx = 0
        for j in range(self.num_antennas_x):
            for i in range(self.num_antennas_y):
                r2 = (x - self.position_x[j]) ** 2 + (self.position_y[i] ** 2) + (z**2)
                r = np.sqrt(r2)
                h = np.exp(
                    -1j
                    * wavenumber
                    * self.distance
                    * (
                        j * np.sin(tx_theta) * np.cos(tx_phi)
                        + i * np.sin(tx_theta) * np.sin(tx_phi)
                    )
                )
                g = np.exp(-1j * wavenumber * r) / r

                reflection_coef = h.conj() * g.conj() / np.abs(h) / np.abs(g)
                cap = self.floque_port.get_capacitance(np.angle(reflection_coef))
                caps.append(cap)
                idx += 1
        return np.array(caps)

    def compute_spherical_wave_energy_density(
        self,
        tx_theta: float,
        tx_phi: float,
        caps: np.ndarray,
        x_z_positions: np.ndarray,
        freq: float,
    ) -> np.ndarray:
        tx_theta = np.deg2rad(tx_theta)
        tx_phi = np.deg2rad(tx_phi)

        wavelength = SPEED_OF_LIGHT / freq
        wavenumber = 2 * np.pi / wavelength

        x = x_z_positions[:, 0]
        z = x_z_positions[:, 1]
        energy_density = np.zeros(len(x_z_positions), dtype=complex)
        idx = 0
        for j in range(self.num_antennas_x):
            for i in range(self.num_antennas_y):
                r2 = (x - self.position_x[j]) ** 2 + (self.position_y[i] ** 2) + (z**2)
                r = np.sqrt(r2)
                h = np.exp(
                    -1j
                    * wavenumber
                    * self.distance
                    * (
                        j * np.sin(tx_theta) * np.cos(tx_phi)
                        + i * np.sin(tx_theta) * np.sin(tx_phi)
                    )
                )
                g = np.exp(-1j * wavenumber * r) / r

                energy_density += (
                    self.floque_port.get_reflection_coeficient(cap=caps[idx]) * h * g
                )
                idx += 1

        return np.abs(energy_density) ** 2


def _compute_temp(
    caps: jax.Array, rad_struct: RisRadiationStructure, freq: float
) -> jax.Array:
    num_rad_ports = rad_struct.get_num_rad_ports()

    diode_r = 1.2
    diode_l = 0.4e-9

    impedances = 1j * 2 * np.pi * freq * diode_l + diode_r / (
        1j * 2 * np.pi * freq * caps / 1e12
    )

    s_t = (impedances - 50) / (impedances + 50)

    s_t_all = jnp.zeros((num_rad_ports,), dtype=complex)
    s_t_all = s_t_all.at[::2].set(s_t)
    s_t_all = s_t_all.at[1::2].set(s_t)

    s_t_rr = jnp.diag(s_t_all)

    x = jnp.linalg.solve(
        jnp.eye(num_rad_ports) - s_t_rr @ rad_struct.s_r_rr, s_t_rr @ rad_struct.s_r_rf
    )

    return x


def compute_physcially_consistent_capacitances(
    target_idx: int, caps: np.ndarray, rad_struct: RisRadiationStructure, freq: float
) -> np.ndarray:
    caps = jnp.array(caps)

    def near_field(c: jax.Array) -> jax.Array:
        return compute_physcially_consistent_energy_denisty(
            c, rad_struct=rad_struct, freq=freq
        )[target_idx]

    val_grad_far_field = jax.jit(jax.value_and_grad(near_field))

    learning_rate = 1e-6
    for idx in range(1000):
        val, phase_grad = val_grad_far_field(caps)
        caps = caps + learning_rate * phase_grad

    return np.array(caps)


def compute_physcially_consistent_energy_denisty(
    caps: jax.Array, rad_struct: RisRadiationStructure, freq: float
) -> jax.Array:
    b_f = jnp.array([1])

    x = _compute_temp(caps=caps, rad_struct=rad_struct, freq=freq)

    g_af_ex = rad_struct.s_r_ff_ex + rad_struct.s_r_fr_ex @ x
    g_af_ey = rad_struct.s_r_ff_ey + rad_struct.s_r_fr_ey @ x
    g_af_ez = rad_struct.s_r_ff_ez + rad_struct.s_r_fr_ez @ x

    g_af_hx = rad_struct.s_r_ff_hx + rad_struct.s_r_fr_hx @ x
    g_af_hy = rad_struct.s_r_ff_hy + rad_struct.s_r_fr_hy @ x
    g_af_hz = rad_struct.s_r_ff_hz + rad_struct.s_r_fr_hz @ x

    r = jnp.abs(g_af_ex @ b_f) ** 2
    r = r + jnp.abs(g_af_ey @ b_f) ** 2
    r = r + jnp.abs(g_af_ez @ b_f) ** 2
    r = r + jnp.abs(g_af_hx @ b_f) ** 2
    r = r + jnp.abs(g_af_hy @ b_f) ** 2
    r = r + jnp.abs(g_af_hz @ b_f) ** 2

    return r


def hfss_ris_parameters32x4(
    freq: float, name: str, tx_theta: int, tx_phi: int
) -> RisRadiationStructure:
    data_dir_near_field = os.path.join(ROOT_PATH, "senf", "ris32x4", f"f{freq/1e9}")
    data_dir_far_field = os.path.join(ROOT_PATH, "senf", "ris32x4")
    s_file_path = os.path.join(ROOT_PATH, "senf", "ris32x4", "ris32x4.s256p")
    return hfss_ris_parameters(
        data_dir_near_field=data_dir_near_field,
        data_dir_far_field=data_dir_far_field,
        s_file_path=s_file_path,
        file_name=name,
        freq=freq,
        tx_theta=tx_theta,
        tx_phi=tx_phi,
    )


def plot_ris(input_coefficent_str: str, used_model_str: str, region_of_interest: str):
    tx_theta = 30
    tx_phi = 0
    rad_struct_rect = hfss_ris_parameters32x4(
        freq=5.8e9, name="rect", tx_theta=tx_theta, tx_phi=tx_phi
    )
    rad_struct_boresight = hfss_ris_parameters32x4(
        freq=5.8e9, name="boresight", tx_theta=tx_theta, tx_phi=tx_phi
    )
    rad_struct_circle = hfss_ris_parameters32x4(
        freq=5.8e9, name="circle", tx_theta=tx_theta, tx_phi=tx_phi
    )
    rad_struct_single = hfss_ris_parameters32x4(
        freq=5.8e9, name="single", tx_theta=tx_theta, tx_phi=tx_phi
    )

    sw_model = SphericalWaveModel(floquet_freq=5.8e9)

    sw_caps = sw_model.compute_spherical_wave_capacitances(
        tx_theta=tx_theta,
        tx_phi=tx_phi,
        x_z_target=rad_struct_single.x_z_idxes[0],
        freq=5.8e9,
    )

    if input_coefficent_str == "pc":
        input_caps = compute_physcially_consistent_capacitances(
            rad_struct=rad_struct_single,
            target_idx=0,
            caps=sw_caps,
            freq=5.8e9,
        )
    elif input_coefficent_str == "sw":
        input_caps = sw_caps
    else:
        raise ValueError()

    if region_of_interest == "rect":
        rad_struct_plot = rad_struct_rect
    elif region_of_interest == "boresight":
        rad_struct_plot = rad_struct_boresight
    elif region_of_interest == "circle":
        rad_struct_plot = rad_struct_circle
    else:
        raise ValueError()

    if used_model_str == "pc":
        energy_density = np.array(
            compute_physcially_consistent_energy_denisty(
                rad_struct=rad_struct_plot, caps=input_caps, freq=5.8e9
            )
        )
    elif used_model_str == "sw":
        energy_density = sw_model.compute_spherical_wave_energy_density(
            tx_theta=tx_theta,
            tx_phi=tx_phi,
            caps=input_caps,
            x_z_positions=rad_struct_plot.x_z_idxes,
            freq=5.8e9,
        )
    else:
        raise ValueError()

    if region_of_interest == "rect":
        plot_rect(
            rad_struct=rad_struct_plot,
            rad_struct_single=rad_struct_single,
            energy_density=energy_density,
        )
    elif region_of_interest == "angle":
        plot_angle(rad_struct=rad_struct_plot, energy_density=energy_density)
    elif region_of_interest == "circle":
        plot_circle(energy_density=energy_density)
    else:
        raise ValueError()


def plot_ris_freq(input_coefficent_str: str, used_model_str: str):
    tx_theta = 30
    tx_phi = 0
    rad_struct_single = hfss_ris_parameters32x4(
        freq=5.8e9, tx_phi=tx_phi, tx_theta=tx_theta, name="single"
    )

    sw_model = SphericalWaveModel(floquet_freq=5.8e9)

    sw_caps = sw_model.compute_spherical_wave_capacitances(
        tx_theta=tx_theta,
        tx_phi=tx_phi,
        x_z_target=rad_struct_single.x_z_idxes[0],
        freq=5.8e9,
    )

    if input_coefficent_str == "pc":
        input_caps = compute_physcially_consistent_capacitances(
            rad_struct=rad_struct_single,
            target_idx=0,
            caps=sw_caps,
            freq=5.8e9,
        )
    elif input_coefficent_str == "sw":
        input_caps = sw_caps
    else:
        raise ValueError()

    energy_densities = []

    fs = np.linspace(4.8e9, 6.8e9, 51)
    for f in fs:
        rad_struct_single_f = hfss_ris_parameters32x4(
            freq=f, tx_phi=tx_phi, tx_theta=tx_theta, name="single"
        )
        if used_model_str == "pc":
            energy_density = np.array(
                compute_physcially_consistent_energy_denisty(
                    rad_struct=rad_struct_single_f, caps=input_caps, freq=f
                )
            )
        elif used_model_str == "sw":
            energy_density = sw_model.compute_spherical_wave_energy_density(
                tx_theta=tx_theta,
                tx_phi=tx_phi,
                caps=input_caps,
                x_z_positions=rad_struct_single_f.x_z_idxes,
                freq=f,
            )
        else:
            raise ValueError()
        energy_densities.append(energy_density)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        layout="constrained",
    )
    ax.plot(fs, 10 * np.log10(energy_densities))
    plt.show()


def plot_rect(
    rad_struct: RisRadiationStructure,
    energy_density: np.ndarray,
    rad_struct_single: RisRadiationStructure,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, layout="constrained")
    dim1, dim2 = find_rectangle_dim(rad_struct.x_z_idxes)

    x_idxes = rad_struct.x_z_idxes[:, 0]
    z_idxes = rad_struct.x_z_idxes[:, 1]
    gain = 10 * np.log10(energy_density)

    x_idxes = x_idxes.reshape((dim1, dim2))
    z_idxes = z_idxes.reshape((dim1, dim2))
    gain = gain.reshape((dim1, dim2))

    im = ax.pcolormesh(z_idxes, x_idxes, gain, shading="auto", rasterized=True)
    im.set_edgecolor("face")

    ax.scatter(
        rad_struct_single.x_z_idxes[0, 1],
        rad_struct_single.x_z_idxes[0, 0],
        c="red",
        linewidth=0.8,
        marker="x",
        s=10,
        zorder=100,
    )

    fig.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_box_aspect(1)

    plt.show()


def plot_angle(rad_struct: RisRadiationStructure, energy_density: np.ndarray):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        layout="constrained",
    )
    ax.plot(np.linalg.norm(rad_struct.x_z_idxes, axis=1), 10 * np.log10(energy_density))
    plt.show()


def plot_circle(energy_density: np.ndarray):
    thetas = np.rad2deg(np.linspace(-np.pi / 2, np.pi / 2, 3143))

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        layout="constrained",
    )
    ax.plot(thetas, 10 * np.log10(energy_density))
    plt.show()
