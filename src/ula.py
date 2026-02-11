import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skrf as rf
from scipy import signal

from src.constants import E_PERMITTIVITY, M_PERMITTIVITY, SPEED_OF_LIGHT
from src.hfss import RadiationStructure, hfss_ula_parameters

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


def get_spherical_wave_antenna_positions(
    num_antennas, reverse_orientation: bool
) -> np.ndarray:
    freq = 10e9
    wavelength = SPEED_OF_LIGHT / freq
    distance = wavelength / 2

    assert num_antennas % 2 == 0

    antenna_positions = (
        distance
        * np.linspace(
            -num_antennas / 2 + 1 / 2, num_antennas / 2 - 1 / 2, num_antennas
        )[::-1]
    )

    if reverse_orientation:
        antenna_positions = antenna_positions[::-1]

    return antenna_positions


def compute_spherical_wave_vector(
    target_x: float, target_z: float, num_antennas: int, method=None
) -> np.ndarray:
    freq = 10e9
    wavelength = SPEED_OF_LIGHT / freq
    wavenumber = 2 * np.pi / wavelength
    offset = 0.001524

    antenna_positions = get_spherical_wave_antenna_positions(
        num_antennas=num_antennas, reverse_orientation=False
    )

    a = np.ones((num_antennas,), dtype=complex)
    for idx, antenna_position in enumerate(antenna_positions):
        r = np.sqrt((target_x - antenna_position) ** 2 + (target_z - offset) ** 2)
        a[idx] = np.exp(-1j * (wavenumber * r)) / r

    vector_peak = a.conj()
    vector_peak = vector_peak / np.linalg.norm(vector_peak)

    vector_phase = np.exp(-1j * np.angle(a))
    vector_phase = vector_phase / np.linalg.norm(vector_phase)

    chebwin = signal.windows.chebwin(num_antennas, at=50)
    vector_cheb = vector_peak * np.array(chebwin)
    vector_cheb = vector_cheb / np.linalg.norm(vector_cheb)

    if method is None:
        return vector_peak
    elif method == "phase":
        return vector_phase
    elif method == "chebwin":
        return vector_cheb
    else:
        raise ValueError("ehhh?")


def compute_spherical_wave_energy_density(
    point_coords: np.ndarray,
    input_vec: np.ndarray,
    num_antennas: int,
    operation_freq: float,
) -> np.ndarray:
    offset = 0.001524

    operation_wavenumber = 2 * np.pi / SPEED_OF_LIGHT * operation_freq

    assert num_antennas % 2 == 0

    antenna_positions = get_spherical_wave_antenna_positions(
        num_antennas=num_antennas, reverse_orientation=False
    )

    a_f = np.zeros(point_coords.shape[0], dtype=complex)
    for idx, antenna_position in enumerate(antenna_positions):
        rr = np.sqrt(
            (point_coords[:, 0] - antenna_position) ** 2
            + (point_coords[:, 1] - offset) ** 2
        )
        a_f += (
            np.exp(-1j * (operation_wavenumber * rr))
            / rr
            * input_vec[idx]
            / np.sqrt(np.pi)
        )
    return np.abs(a_f) ** 2


def compute_spherical_wave_vector_obstacle(
    target_x: float,
    target_z: float,
    num_antennas: int,
    operation_freq: float,
) -> np.ndarray:
    offset = 0.001524
    head_x = 0.8849999999999999
    head_z = 1.5348649646984565
    head_r = 0.1

    operation_wavenumber = 2 * np.pi / SPEED_OF_LIGHT * operation_freq

    antenna_positions = get_spherical_wave_antenna_positions(
        num_antennas=num_antennas, reverse_orientation=True
    )

    result = np.ones((num_antennas,), dtype=complex)
    for idx, antenna_position in enumerate(antenna_positions):
        x0 = head_x
        y0 = head_z
        r = head_r

        x1 = target_x
        y1 = target_z

        x2 = antenna_position
        y2 = offset

        cx = x0
        cy = x1
        ax = x1 - x0
        ay = y1 - y0
        bx = x2 - cx
        by = y2 - cy
        a = (bx - ax) ** 2 + (by - ay) ** 2
        b = 2 * (ax * (bx - ax) + ay * (by - ay))
        c = ax**2 + ay**2 - r**2
        disc = b**2 - 4 * a * c
        sqrtdisc = np.sqrt(disc)
        t1 = (-b + sqrtdisc) / (2 * a)
        t2 = (-b - sqrtdisc) / (2 * a)
        mask = (((0 < t1) & (t1 < 1)) | ((0 < t2) & (t2 < 1))) & (disc > 0)

        rr = np.sqrt((target_x - antenna_position) ** 2 + (target_z - offset) ** 2)

        result[idx] = (1 - mask) * (
            np.exp(-1j * (operation_wavenumber * rr)) / rr / np.sqrt(np.pi)
        )

    vector_peak = result.conj()
    vector_peak = vector_peak / np.linalg.norm(vector_peak)
    return vector_peak


def compute_spherical_wave_energy_density_obstacle(
    point_coords: np.ndarray,
    input_vec: np.ndarray,
    num_antennas: int,
    operation_freq: float,
) -> np.ndarray:
    offset = 0.001524
    head_x = 0.8849999999999999
    head_z = 1.5348649646984565
    head_r = 0.1

    operation_wavenumber = 2 * np.pi / SPEED_OF_LIGHT * operation_freq

    antenna_positions = get_spherical_wave_antenna_positions(
        num_antennas=num_antennas, reverse_orientation=True
    )

    a_f = np.zeros(point_coords.shape[0], dtype=complex)
    for idx, antenna_position in enumerate(antenna_positions):
        x0 = head_x
        y0 = head_z
        r = head_r

        x1 = point_coords[:, 0]
        y1 = point_coords[:, 1]

        x2 = antenna_position
        y2 = offset

        cx = x0
        cy = x1
        ax = x1 - x0
        ay = y1 - y0
        bx = x2 - cx
        by = y2 - cy
        a = (bx - ax) ** 2 + (by - ay) ** 2
        b = 2 * (ax * (bx - ax) + ay * (by - ay))
        c = ax**2 + ay**2 - r**2
        disc = b**2 - 4 * a * c
        sqrtdisc = np.sqrt(disc)
        t1 = (-b + sqrtdisc) / (2 * a)
        t2 = (-b - sqrtdisc) / (2 * a)
        mask = (((0 < t1) & (t1 < 1)) | ((0 < t2) & (t2 < 1))) & (disc > 0)

        rr = np.sqrt(
            (point_coords[:, 0] - antenna_position) ** 2
            + (point_coords[:, 1] - offset) ** 2
        )
        a_f += (1 - mask) * (
            np.exp(-1j * (operation_wavenumber * rr))
            / rr
            * input_vec[idx]
            / np.sqrt(np.pi)
        )
    return np.abs(a_f) ** 2


def get_tuning_network(
    rad_struct: RadiationStructure,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_rad_ports = rad_struct.get_num_rad_ports()
    num_tx_ports = num_rad_ports
    ref_z0 = 50

    s_t_tt = np.zeros((num_rad_ports, num_rad_ports))
    s_t_tr = np.eye(num_rad_ports, num_rad_ports)
    s_t_rt = np.eye(num_rad_ports, num_rad_ports)
    s_t_rr = np.zeros((num_rad_ports, num_rad_ports))

    tx_impedances = np.eye(num_tx_ports) * 50
    temp_tx = np.linalg.inv(tx_impedances + ref_z0 * np.eye(num_tx_ports))
    k_tx = temp_tx * np.sqrt(ref_z0)
    s_rf_tx = temp_tx @ (tx_impedances - ref_z0 * np.eye(num_tx_ports))

    s_rf = s_rf_tx

    return s_t_tt, s_t_rt, s_t_tr, s_t_rr, s_rf, k_tx


def _compute_g_r(
    rad_struct: RadiationStructure,
    s_t_tt: np.ndarray,
    s_t_rt: np.ndarray,
    s_t_tr: np.ndarray,
    s_t_rr: np.ndarray,
    s_rf: np.ndarray,
    k_tx: np.ndarray,
) -> np.ndarray:

    num_rad_ports = rad_struct.get_num_rad_ports()
    num_tun_ports = s_t_tt.shape[0]

    l2 = s_t_rr @ rad_struct.s_r_rr
    l1 = s_rf @ s_t_tt
    l3 = (
        s_rf
        @ s_t_tr
        @ rad_struct.s_r_rr
        @ np.linalg.solve(np.eye(num_rad_ports) - l2, s_t_rt)
    )
    x = s_t_rt @ np.linalg.solve(np.eye(num_tun_ports) - l1 - l3, k_tx)
    g_v_tick = np.linalg.solve(np.eye(num_rad_ports) - l2, x)

    return g_v_tick


def compute_g_r(rad_struct: RadiationStructure) -> np.ndarray:
    s_t_tt, s_t_rt, s_t_tr, s_t_rr, s_rf, k_tx = get_tuning_network(
        rad_struct=rad_struct
    )

    return _compute_g_r(
        rad_struct=rad_struct,
        s_t_tt=s_t_tt,
        s_t_rt=s_t_rt,
        s_t_tr=s_t_tr,
        s_t_rr=s_t_rr,
        s_rf=s_rf,
        k_tx=k_tx,
    )


def compute_physcially_consistent_vector(
    rad_struct: RadiationStructure, target_idx: int
) -> np.ndarray:
    num_antennas = rad_struct.get_num_rad_ports()
    target_s_e = np.stack(
        [
            rad_struct.s_ex[target_idx],
            rad_struct.s_ey[target_idx],
            rad_struct.s_ez[target_idx],
        ]
    )
    target_s_h = np.stack(
        [
            rad_struct.s_hx[target_idx],
            rad_struct.s_hy[target_idx],
            rad_struct.s_hz[target_idx],
        ]
    )

    g_a_r = compute_g_r(rad_struct=rad_struct)

    g_e = target_s_e @ g_a_r
    g_h = target_s_h @ g_a_r

    g = g_e.conj().T @ g_e + g_h.conj().T @ g_h

    pa_impedances = np.eye(num_antennas) * 50
    b = np.real(np.linalg.inv(pa_impedances))

    eigenvalues, eigenvectors = scipy.linalg.eigh(a=g, b=b)

    res = eigenvectors[:, -1] / np.linalg.norm(eigenvectors[:, -1])

    return res


def compute_physcially_consistent_energy_density(
    rad_struct: RadiationStructure, input_vec: np.ndarray
) -> np.ndarray:
    g_a_r = compute_g_r(rad_struct=rad_struct)

    e_field = (
        np.stack([rad_struct.s_ex, rad_struct.s_ey, rad_struct.s_ez])
        @ g_a_r
        @ input_vec
    )
    h_field = (
        np.stack([rad_struct.s_hx, rad_struct.s_hy, rad_struct.s_hz])
        @ g_a_r
        @ input_vec
    )

    e_energy = np.linalg.vector_norm(e_field, axis=0) ** 2
    h_energy = np.linalg.vector_norm(h_field, axis=0) ** 2

    energy_density = e_energy + h_energy

    return energy_density * np.sqrt(E_PERMITTIVITY * M_PERMITTIVITY)


def find_rectangle_dim(x_z_idxes: np.ndarray) -> tuple[int, int]:
    jj = x_z_idxes[:, 1][0]
    tol = 1e-7
    dim2 = ((jj - tol < x_z_idxes[:, 1]) & (x_z_idxes[:, 1] < jj + tol)).sum()
    dim1 = len(x_z_idxes) // dim2

    return dim1, dim2


def hfss_ula_parameters128(freq: float, name: str) -> RadiationStructure:
    data_dir = os.path.join(ROOT_PATH, "senf", "array128", f"f{freq/1e9}")
    s_file_path = os.path.join(ROOT_PATH, "senf", "array128", "array128.s128p")
    return hfss_ula_parameters(
        data_dir=data_dir,
        s_file_path=s_file_path,
        file_name=name,
        freq=freq,
    )


def hfss_ula_parameters_obstacle128(freq: float, name: str) -> RadiationStructure:
    data_dir = os.path.join(ROOT_PATH, "senf", "array128_obstacle")
    s_file_path = os.path.join(data_dir, "array128_obstacle.s128p")
    return hfss_ula_parameters(
        data_dir=data_dir,
        s_file_path=s_file_path,
        file_name=name,
        freq=freq,
    )


def plot_ula(input_vec_str: str, used_model_str: str, region_of_interest: str):
    rad_struct_rect = hfss_ula_parameters128(freq=10e9, name="rect")
    rad_struct_angle = hfss_ula_parameters128(freq=10e9, name="angle")
    rad_struct_circle = hfss_ula_parameters128(freq=10e9, name="circle")
    rad_struct_single = hfss_ula_parameters128(freq=10e9, name="single")

    if input_vec_str == "pc":
        input_vec = compute_physcially_consistent_vector(
            rad_struct=rad_struct_single, target_idx=0
        )
    elif input_vec_str == "sw":
        input_vec = compute_spherical_wave_vector(
            target_x=rad_struct_single.x_z_idxes[0, 0],
            target_z=rad_struct_single.x_z_idxes[0, 1],
            num_antennas=rad_struct_rect.get_num_rad_ports(),
        )
    else:
        raise ValueError()

    if region_of_interest == "rect":
        rad_struct_plot = rad_struct_rect
    elif region_of_interest == "angle":
        rad_struct_plot = rad_struct_angle
    elif region_of_interest == "circle":
        rad_struct_plot = rad_struct_circle
    else:
        raise ValueError()

    if used_model_str == "pc":
        energy_density = compute_physcially_consistent_energy_density(
            rad_struct=rad_struct_plot, input_vec=input_vec
        )
    elif used_model_str == "sw":
        energy_density = compute_spherical_wave_energy_density(
            point_coords=rad_struct_plot.x_z_idxes,
            input_vec=input_vec,
            num_antennas=rad_struct_plot.get_num_rad_ports(),
            operation_freq=10e9,
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
        plot_circle(rad_struct=rad_struct_plot, energy_density=energy_density)
    else:
        raise ValueError()


def plot_ula_freq(input_vec_str: str, used_model_str: str):
    rad_struct_single = hfss_ula_parameters128(freq=10e9, name="single")

    if input_vec_str == "pc":
        input_vec = compute_physcially_consistent_vector(
            rad_struct=rad_struct_single, target_idx=0
        )
    elif input_vec_str == "sw":
        input_vec = compute_spherical_wave_vector(
            target_x=rad_struct_single.x_z_idxes[0, 0],
            target_z=rad_struct_single.x_z_idxes[0, 1],
            num_antennas=rad_struct_single.get_num_rad_ports(),
        )
    else:
        raise ValueError()

    energy_densities = []

    fs = np.linspace(8e9, 12e9, 101)
    for f in fs:
        rad_struct_single_f = hfss_ula_parameters128(freq=f, name="single")
        if used_model_str == "pc":
            energy_density = compute_physcially_consistent_energy_density(
                rad_struct=rad_struct_single_f, input_vec=input_vec
            )
        elif used_model_str == "sw":
            energy_density = compute_spherical_wave_energy_density(
                point_coords=rad_struct_single_f.x_z_idxes,
                input_vec=input_vec,
                num_antennas=rad_struct_single_f.get_num_rad_ports(),
                operation_freq=f,
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


def plot_ula_obstacle(input_vec_str: str, used_model_str: str, region_of_interest: str):
    rad_struct_rect = hfss_ula_parameters_obstacle128(freq=10e9, name="rect")
    rad_struct_single = hfss_ula_parameters_obstacle128(freq=10e9, name="single")
    rad_struct_angle = hfss_ula_parameters_obstacle128(freq=10e9, name="angle")

    if input_vec_str == "pc":
        input_vec = compute_physcially_consistent_vector(
            rad_struct=rad_struct_single, target_idx=0
        )
    elif input_vec_str == "sw":
        input_vec = compute_spherical_wave_vector_obstacle(
            target_x=rad_struct_single.x_z_idxes[0, 0],
            target_z=rad_struct_single.x_z_idxes[0, 1],
            num_antennas=rad_struct_rect.get_num_rad_ports(),
            operation_freq=10e9,
        )
    else:
        raise ValueError()

    if region_of_interest == "rect":
        rad_struct_plot = rad_struct_rect
    elif region_of_interest == "angle":
        rad_struct_plot = rad_struct_angle
    else:
        raise ValueError()

    if used_model_str == "pc":
        energy_density = compute_physcially_consistent_energy_density(
            rad_struct=rad_struct_plot, input_vec=input_vec
        )
    elif used_model_str == "sw":
        energy_density = compute_spherical_wave_energy_density_obstacle(
            point_coords=rad_struct_plot.x_z_idxes,
            input_vec=input_vec,
            num_antennas=rad_struct_plot.get_num_rad_ports(),
            operation_freq=10e9,
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
        plot_circle(rad_struct=rad_struct_plot, energy_density=energy_density)
    else:
        raise ValueError()


def plot_rect(
    rad_struct: RadiationStructure,
    energy_density: np.ndarray,
    rad_struct_single: RadiationStructure,
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


def plot_angle(rad_struct: RadiationStructure, energy_density: np.ndarray):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        layout="constrained",
    )
    ax.plot(np.linalg.norm(rad_struct.x_z_idxes, axis=1), 10 * np.log10(energy_density))
    plt.show()


def plot_circle(rad_struct: RadiationStructure, energy_density: np.ndarray):
    thetas = np.rad2deg(np.linspace(-np.pi / 2, np.pi / 2, 3143))

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        layout="constrained",
    )
    ax.plot(thetas, 10 * np.log10(energy_density))
    plt.show()
