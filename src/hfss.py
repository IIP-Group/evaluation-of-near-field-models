import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import skrf as rf

from src.constants import FREE_WAVE_IMPEDANCE, SPEED_OF_LIGHT

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

NFD_HEADER = "#Index, X, Y, Z, Ex(real, imag), Ey(real, imag), Ez(real, imag), Hx(real, imag), Hy(real, imag), Hz(real, imag)"
NFD_NAMES = [
    "idx",
    "x",
    "y",
    "z",
    "Ex_real",
    "Ex_imag",
    "Ey_real",
    "Ey_imag",
    "Ez_real",
    "Ez_imag",
    "Hx_real",
    "Hx_imag",
    "Hy_real",
    "Hy_imag",
    "Hz_real",
    "Hz_imag",
]

NFD_DTYPES = {
    "idx": int,
    "x": float,
    "y": float,
    "z": float,
    "Ex_real": float,
    "Ex_imag": float,
    "Ey_real": float,
    "Ey_imag": float,
    "Ez_real": float,
    "Ez_imag": float,
    "Hx_real": float,
    "Hx_imag": float,
    "Hy_real": float,
    "Hy_imag": float,
    "Hz_real": float,
    "Hz_imag": float,
}


@dataclass
class RadiationStructure:
    s_r_rr: np.ndarray
    x_z_idxes: np.ndarray
    s_ex: np.ndarray
    s_ey: np.ndarray
    s_ez: np.ndarray
    s_hx: np.ndarray
    s_hy: np.ndarray
    s_hz: np.ndarray

    def get_num_rad_ports(self) -> int:
        return self.s_r_rr.shape[0]


@dataclass
class RisRadiationStructure:
    s_r_rr: np.ndarray
    x_z_idxes: np.ndarray
    s_r_rf: np.ndarray
    s_r_fr_ex: np.ndarray
    s_r_fr_ey: np.ndarray
    s_r_fr_ez: np.ndarray
    s_r_fr_hx: np.ndarray
    s_r_fr_hy: np.ndarray
    s_r_fr_hz: np.ndarray
    s_r_ff_ex: np.ndarray
    s_r_ff_ey: np.ndarray
    s_r_ff_ez: np.ndarray
    s_r_ff_hx: np.ndarray
    s_r_ff_hy: np.ndarray
    s_r_ff_hz: np.ndarray

    def get_num_rad_ports(self) -> int:
        return self.s_r_rr.shape[0]


def read_touchstone_file(file: str, freq: float) -> np.ndarray:
    rf_net = rf.Network(file=file)
    idx = (np.abs(rf_net.f - freq)).argmin()
    if np.abs(rf_net.f[idx] - freq) > 0.1:
        raise Exception(f"frequency {freq} not file")

    return np.array(rf_net.s[idx])


def str_to_complex(x):
    return complex(x.replace(" ", "").replace("i", "j"))


def read_far_field_csv(path: str, freq: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[np.abs(df["Freq [GHz]"] * 1e9 - freq) < 1]
    return df


def _extract_far_field(
    phi_theta_idxes: np.ndarray,
    num_ports: int,
    name: str,
    file_name: str,
    data_dir: str,
    freq: float,
) -> np.ndarray:
    num_samples = phi_theta_idxes.shape[0]
    s_r_fr_c = np.zeros((num_samples, num_ports), dtype=complex)

    for idx in range(num_ports):
        df = read_far_field_csv(
            os.path.join(data_dir, f"rE{file_name}{idx}.csv"), freq=freq
        )

        i_phi_theta_idxes = np.array(df[["Phi [deg]", "Theta [deg]"]])
        i_phi_theta_idxes[:, 0] = i_phi_theta_idxes[:, 0] % 360
        assert np.all(i_phi_theta_idxes == phi_theta_idxes)
        if f"rE{name} [V]" in df.columns.values:
            e_field = np.array(df[f"rE{name} [V]"].apply(str_to_complex))
        elif f"rE{name} [mV]" in df.columns.values:
            e_field = np.array(df[f"rE{name} [mV]"].apply(str_to_complex)) / 1000.0
        elif f"rE{name} [uV]" in df.columns.values:
            e_field = np.array(df[f"rE{name} [uV]"].apply(str_to_complex)) / 1000_000.0
        elif f"rE{name} [pV]" in df.columns.values:
            e_field = np.array(df[f"rE{name} [pV]"].apply(str_to_complex)) * 1e-12
        elif f"rE{name} [fV]" in df.columns.values:
            e_field = np.array(df[f"rE{name} [fV]"].apply(str_to_complex)) * 1e-15
        else:
            print(name, df.columns.values, file_name)
            raise Exception("what?")

        s_r_fr_c[:, idx] = e_field

    return s_r_fr_c


def _ndf_to_df(path: str) -> pd.DataFrame:
    with open(path) as f:
        header = f.readline().strip()
        assert header == NFD_HEADER
        f.readline()
        f.readline()
        df = pd.read_csv(f, header=None, names=NFD_NAMES)
    return df


def _extract_near_field(
    path: str, x_z_idxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = _ndf_to_df(path)
    assert np.all(np.array(df[["x", "z"]]) == x_z_idxes)
    y_idxes = df["y"]
    assert np.allclose(y_idxes, np.zeros_like(y_idxes))

    ex = (df["Ex_real"] + 1j * df["Ex_imag"]).to_numpy()
    ey = (df["Ey_real"] + 1j * df["Ey_imag"]).to_numpy()
    ez = (df["Ez_real"] + 1j * df["Ez_imag"]).to_numpy()

    hx = (df["Hx_real"] + 1j * df["Hx_imag"]).to_numpy()
    hy = (df["Hy_real"] + 1j * df["Hy_imag"]).to_numpy()
    hz = (df["Hz_real"] + 1j * df["Hz_imag"]).to_numpy()

    return ex, ey, ez, hx, hy, hz


def _extract_far_field2near_field(
    num_ports: int, data_dir: str, file_name: str
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    df_template = _ndf_to_df(
        os.path.join(data_dir, f"near_field_plane_{file_name}0.nfd")
    )

    x_z_idxes = np.array(df_template[["x", "z"]])
    y_idxes = df_template["y"]
    assert np.allclose(y_idxes, np.zeros_like(y_idxes))

    num_samples = x_z_idxes.shape[0]

    s_ex = np.zeros((num_samples, num_ports), dtype=complex)
    s_ey = np.zeros((num_samples, num_ports), dtype=complex)
    s_ez = np.zeros((num_samples, num_ports), dtype=complex)
    s_hx = np.zeros((num_samples, num_ports), dtype=complex)
    s_hy = np.zeros((num_samples, num_ports), dtype=complex)
    s_hz = np.zeros((num_samples, num_ports), dtype=complex)

    for idx in range(num_ports):
        ex, ey, ez, hx, hy, hz = _extract_near_field(
            os.path.join(data_dir, f"near_field_plane_{file_name}{idx}.nfd"),
            x_z_idxes=x_z_idxes,
        )
        s_ex[:, idx] = ex
        s_ey[:, idx] = ey
        s_ez[:, idx] = ez

        s_hx[:, idx] = hx
        s_hy[:, idx] = hy
        s_hz[:, idx] = hz

    return x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz


def _extract_port2near_field(
    num_ports: int,
    data_dir: str,
    file_name: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    time_stamp = max(
        [
            os.path.getmtime(os.path.join(data_dir, f"near_field_{file_name}{idx}.nfd"))
            for idx in range(num_ports)
        ]
    )
    save_path_name = re.sub(r"[^\w_. -]", "_", data_dir)
    cache_path = os.path.join(
        ROOT_PATH, "cache", f"{save_path_name}cache{file_name}{time_stamp}.npz"
    )
    if os.path.exists(cache_path):
        npzfile = np.load(cache_path)
        x_z_idxes = npzfile["x_z_idxes"]
        s_ex = npzfile["s_ex"]
        s_ey = npzfile["s_ey"]
        s_ez = npzfile["s_ez"]
        s_hx = npzfile["s_hx"]
        s_hy = npzfile["s_hy"]
        s_hz = npzfile["s_hz"]
        return x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz

    df_template = _ndf_to_df(os.path.join(data_dir, f"near_field_{file_name}0.nfd"))

    x_z_idxes = np.array(df_template[["x", "z"]])
    y_idxes = df_template["y"]
    assert np.allclose(y_idxes, np.zeros_like(y_idxes))

    num_samples = x_z_idxes.shape[0]

    s_ex = np.zeros((num_samples, num_ports), dtype=complex)
    s_ey = np.zeros((num_samples, num_ports), dtype=complex)
    s_ez = np.zeros((num_samples, num_ports), dtype=complex)
    s_hx = np.zeros((num_samples, num_ports), dtype=complex)
    s_hy = np.zeros((num_samples, num_ports), dtype=complex)
    s_hz = np.zeros((num_samples, num_ports), dtype=complex)

    for idx in range(num_ports):
        ex, ey, ez, hx, hy, hz = _extract_near_field(
            os.path.join(data_dir, f"near_field_{file_name}{idx}.nfd"),
            x_z_idxes=x_z_idxes,
        )

        s_ex[:, idx] = ex
        s_ey[:, idx] = ey
        s_ez[:, idx] = ez

        s_hx[:, idx] = hx
        s_hy[:, idx] = hy
        s_hz[:, idx] = hz

    np.savez(
        cache_path,
        x_z_idxes=x_z_idxes,
        s_ex=s_ex,
        s_ey=s_ey,
        s_ez=s_ez,
        s_hx=s_hx,
        s_hy=s_hy,
        s_hz=s_hz,
    )

    return x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz


def extract_port2far_field(
    phi_theta_idxes: np.ndarray,
    num_ports: int,
    name: str,
    file_name: str,
    data_dir: str,
    freq: float,
) -> np.ndarray:
    e_field = _extract_far_field(
        phi_theta_idxes=phi_theta_idxes,
        num_ports=num_ports,
        name=name,
        file_name=file_name,
        data_dir=data_dir,
        freq=freq,
    )

    s_r_fr_c = e_field / np.sqrt(2 * FREE_WAVE_IMPEDANCE)

    return s_r_fr_c


def extract_far_field2far_field(
    phi_theta_idxes: np.ndarray,
    num_ports: int,
    name: str,
    file_name: str,
    data_dir: str,
    freq: float,
) -> np.ndarray:
    e_field = _extract_far_field(
        phi_theta_idxes=phi_theta_idxes,
        num_ports=num_ports,
        name=name,
        file_name=file_name,
        data_dir=data_dir,
        freq=freq,
    )

    k = 2 * np.pi / (SPEED_OF_LIGHT / freq)
    s_r_fr_c = e_field * k / (2 * np.pi / 1j)

    return s_r_fr_c


def extract_port2near_field(
    num_ports: int,
    data_dir: str,
    file_name: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz = _extract_port2near_field(
        num_ports=num_ports, data_dir=data_dir, file_name=file_name
    )

    s_ex /= np.sqrt(FREE_WAVE_IMPEDANCE)
    s_ey /= np.sqrt(FREE_WAVE_IMPEDANCE)
    s_ez /= np.sqrt(FREE_WAVE_IMPEDANCE)
    s_hx *= np.sqrt(FREE_WAVE_IMPEDANCE)
    s_hy *= np.sqrt(FREE_WAVE_IMPEDANCE)
    s_hz *= np.sqrt(FREE_WAVE_IMPEDANCE)

    s_ex /= np.sqrt(2)
    s_ey /= np.sqrt(2)
    s_ez /= np.sqrt(2)
    s_hx /= np.sqrt(2)
    s_hy /= np.sqrt(2)
    s_hz /= np.sqrt(2)

    return x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz


def extract_far_field2near_field(
    num_ports: int,
    data_dir: str,
    freq: float,
    file_name: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz = _extract_far_field2near_field(
        num_ports=num_ports,
        data_dir=data_dir,
        file_name=file_name,
    )

    k = 2 * np.pi / (SPEED_OF_LIGHT / freq)

    s_ex *= k / (2 * np.pi / 1j)
    s_ey *= k / (2 * np.pi / 1j)
    s_ez *= k / (2 * np.pi / 1j)
    s_hx *= k / (2 * np.pi / 1j)
    s_hy *= k / (2 * np.pi / 1j)
    s_hz *= k / (2 * np.pi / 1j)

    return x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz


def hfss_ula_parameters(
    data_dir: str, s_file_path: str, file_name: str, freq: float
) -> RadiationStructure:
    s_r_rr = read_touchstone_file(s_file_path, freq=freq)

    num_ports = s_r_rr.shape[0]

    x_z_idxes, s_ex, s_ey, s_ez, s_hx, s_hy, s_hz = extract_port2near_field(
        num_ports=num_ports, data_dir=data_dir, file_name=file_name
    )

    return RadiationStructure(
        s_r_rr=s_r_rr,
        x_z_idxes=np.array(x_z_idxes),
        s_ex=np.array(s_ex),
        s_ey=np.array(s_ey),
        s_ez=np.array(s_ez),
        s_hx=np.array(s_hx),
        s_hy=np.array(s_hy),
        s_hz=np.array(s_hz),
    )


def hfss_ris_parameters(
    data_dir_near_field: str,
    data_dir_far_field: str,
    s_file_path: str,
    tx_phi: int,
    tx_theta: int,
    freq: float,
    file_name: str,
) -> RisRadiationStructure:
    s_r_rr = read_touchstone_file(s_file_path, freq=freq)

    num_ports = s_r_rr.shape[0]
    df_far_structure = read_far_field_csv(
        os.path.join(data_dir_far_field, "rEPhi_0.csv"), freq=freq
    )
    df_far_structure = df_far_structure[
        np.abs(df_far_structure["Freq [GHz]"] * 1e9 - freq) < 1
    ]
    phi_theta_idxes = np.array(df_far_structure[["Phi [deg]", "Theta [deg]"]])
    phi_theta_idxes[:, 0] = phi_theta_idxes[:, 0] % 360

    selected_phis = list(range(0, 360, 90))
    selected_thetas = [30]

    selected_idx = []
    for theta in selected_thetas:
        for phi in selected_phis:
            x = np.nonzero(
                (phi_theta_idxes[:, 0] == phi) & (phi_theta_idxes[:, 1] == theta)
            )[0][0]
            selected_idx.append(x)
    selected_idx = np.array(selected_idx)

    s_r_fr_theta = extract_port2far_field(
        phi_theta_idxes=phi_theta_idxes,
        num_ports=num_ports,
        name="Theta",
        file_name="Theta_",
        data_dir=data_dir_far_field,
        freq=freq,
    )
    s_r_fr_phi = extract_port2far_field(
        phi_theta_idxes=phi_theta_idxes,
        num_ports=num_ports,
        name="Phi",
        file_name="Phi_",
        data_dir=data_dir_far_field,
        freq=freq,
    )

    x_z_idxes_ff, s_r_ff_ex, s_r_ff_ey, s_r_ff_ez, s_r_ff_hx, s_r_ff_hy, s_r_ff_hz = (
        extract_far_field2near_field(
            num_ports=1, data_dir=data_dir_near_field, freq=freq, file_name=file_name
        )
    )

    x_z_idxes, s_r_fr_ex, s_r_fr_ey, s_r_fr_ez, s_r_fr_hx, s_r_fr_hy, s_r_fr_hz = (
        extract_port2near_field(
            num_ports=num_ports, data_dir=data_dir_near_field, file_name=file_name
        )
    )

    assert np.allclose(x_z_idxes, x_z_idxes_ff)

    x = np.nonzero(
        (phi_theta_idxes[:, 0] == tx_phi) & (phi_theta_idxes[:, 1] == tx_theta)
    )[0][0]

    s_r_rf = s_r_fr_phi[[x]].T

    return RisRadiationStructure(
        s_r_rr=s_r_rr,
        x_z_idxes=np.array(x_z_idxes),
        s_r_rf=np.array(s_r_rf),
        s_r_ff_ex=np.array(s_r_ff_ex),
        s_r_ff_ey=np.array(s_r_ff_ey),
        s_r_ff_ez=np.array(s_r_ff_ez),
        s_r_ff_hx=np.array(s_r_ff_hx),
        s_r_ff_hy=np.array(s_r_ff_hy),
        s_r_ff_hz=np.array(s_r_ff_hz),
        s_r_fr_ex=np.array(s_r_fr_ex),
        s_r_fr_ey=np.array(s_r_fr_ey),
        s_r_fr_ez=np.array(s_r_fr_ez),
        s_r_fr_hx=np.array(s_r_fr_hx),
        s_r_fr_hy=np.array(s_r_fr_hy),
        s_r_fr_hz=np.array(s_r_fr_hz),
    )
