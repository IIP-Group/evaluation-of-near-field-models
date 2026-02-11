# ----------------------------------------------
# Script Recorded by Ansys Electronics Desktop Version 2022.1.0
# 10:16:25  Sep 11, 2025
# ----------------------------------------------
import os

import ScriptEnv

X = 2
N = 32
M = 4

DIR = "<full_path_to_directory>"
PROJECT_NAME = "evaluation_of_near_field_models"
DESGIN_NAME = "ris32x4"


def linspace(x1, x2, nstep):
    step = (x2 - x1) / (nstep - 1)
    return [round(x1 + (i * step), 8) for i in range(0, nstep)]


def edit_sources_array(power_idx):
    sources = [
        [
            "FieldType:=",
            "ScatteredFields",
            "IncludePortPostProcessing:=",
            False,
            "SpecifySystemPower:=",
            False,
        ],
    ]

    c = 0
    for idx1 in range(1, N + 1):
        for idx2 in range(1, M + 1):
            for idx3 in range(1, X + 1):
                power = 1 if power_idx == c else 0
                c += 1
                name = (
                    "A1["
                    + str(idx1)
                    + ","
                    + str(idx2)
                    + "]ris_component1_"
                    + str(idx3)
                    + ":1"
                )
                sources.append(
                    ["Name:=", name, "Magnitude:=", str(power) + "W", "Phase:=", "0deg"]
                )

    sources.append(
        ["Name:=", "IncPWave1", "Magnitude:=", "0V_per_meter", "Phase:=", "0deg"]
    )
    return sources


def edit_sources_plane_wave():
    sources = [
        [
            "FieldType:=",
            "ScatteredFields",
            "IncludePortPostProcessing:=",
            False,
            "SpecifySystemPower:=",
            False,
        ],
    ]

    for idx1 in range(1, N + 1):
        for idx2 in range(1, M + 1):
            for idx3 in range(1, X + 1):
                power = 0
                name = (
                    "A1["
                    + str(idx1)
                    + ","
                    + str(idx2)
                    + "]ris_component1_"
                    + str(idx3)
                    + ":1"
                )
                sources.append(
                    ["Name:=", name, "Magnitude:=", str(power) + "W", "Phase:=", "0deg"]
                )

    sources.append(
        ["Name:=", "IncPWave1", "Magnitude:=", "1V_per_meter", "Phase:=", "0deg"]
    )
    return sources


def export_near_field(path, point_name, f):
    oModule3 = oDesign.GetModule("RadField")
    oModule3.ExportFieldsToFile(
        [
            "ExportFileName:=",
            path,
            "SetupName:=",
            point_name,
            "IntrinsicVariationKey:=",
            "Freq='" + str(f) + "GHz'",
            "DesignVariationKey:=",
            "",
            "SolutionName:=",
            "Setup1: Sweep",
            "Quantity:=",
            "E",
            "ExportInLocal:=",
            False,
            "ExportANDFile:=",
            False,
        ]
    )


print("starting")

ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.SetActiveProject(PROJECT_NAME)
oDesign = oProject.SetActiveDesign(DESGIN_NAME)

oModule = oDesign.GetModule("Solutions")
oModule.ExportNetworkData(
    "",
    ["Setup1: Sweep"],
    3,
    os.path.join(DIR, DESGIN_NAME + ".s" + str(N * M * X) + "p"),
    ["All"],
    True,
    50,
    "S",
    -1,
    0,
    15,
    True,
    True,
    False,
)


for idx in range(0, N * M * X):
    oModule1 = oDesign.GetModule("Solutions")
    oModule1.EditSources(edit_sources_array(idx))

    oModule2 = oDesign.GetModule("ReportSetup")
    oModule2.UpdateReports(["rEPhi"])
    oModule2.ExportToFile(
        "rEPhi", os.path.join(DIR, "rEPhi_" + str(idx) + ".csv"), False
    )
    oModule2.UpdateReports(["rETheta"])
    oModule2.ExportToFile(
        "rETheta", os.path.join(DIR, "rETheta_" + str(idx) + ".csv"), False
    )

    f = 5.8
    dir_path = os.path.join(DIR, "f" + str(f))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    export_near_field(
        os.path.join(dir_path, "near_field_cirlce" + str(idx) + ".nfd"), "circle", f
    )
    export_near_field(
        os.path.join(dir_path, "near_field_boresight" + str(idx) + ".nfd"),
        "boresight",
        f,
    )
    export_near_field(
        os.path.join(dir_path, "near_field_rect" + str(idx) + ".nfd"), "rect", f
    )

    for f in linspace(4.8, 6.8, 51):
        dir_path = os.path.join(DIR, "f" + str(f))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        export_near_field(
            os.path.join(dir_path, "near_field_single" + str(idx) + ".nfd"), "single", f
        )


oModule1 = oDesign.GetModule("Solutions")
oModule1.EditSources(edit_sources_plane_wave())

oModule2 = oDesign.GetModule("ReportSetup")
oModule2.UpdateReports(["rEPhi"])
oModule2.ExportToFile("rEPhi", os.path.join(DIR, "rEPhi_" + "plane_0" + ".csv"), False)
oModule2.UpdateReports(["rETheta"])
oModule2.ExportToFile(
    "rETheta", os.path.join(DIR, "rETheta_" + "plane_0" + ".csv"), False
)

f = 5.8
dir_path = os.path.join(DIR, "f" + str(f))
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
export_near_field(
    os.path.join(dir_path, "near_field_plane_cirlce" + str(0) + ".nfd"), "circle", f
)
export_near_field(
    os.path.join(dir_path, "near_field_plane_boresight" + str(0) + ".nfd"),
    "boresight",
    f,
)
export_near_field(
    os.path.join(dir_path, "near_field_plane_rect" + str(0) + ".nfd"), "rect", f
)

for f in linspace(4.8, 6.8, 51):
    dir_path = os.path.join(DIR, "f" + str(f))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    export_near_field(
        os.path.join(dir_path, "near_field_plane_single" + str(0) + ".nfd"), "single", f
    )
