import os

import ScriptEnv

N = 128
DIR = "<full_path_to_directory>"
PROJECT_NAME = "evaluation_of_near_field_models"
DESGIN_NAME = "array128"


def linspace(x1, x2, nstep):
    step = (x2 - x1) / (nstep - 1)
    return [round(x1 + (i * step), 8) for i in range(0, nstep)]


def edit_sources_array(power_idx):
    sources = [["IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False]]
    for idx2 in range(1, N + 1):
        power = 1 if power_idx == idx2 else 0
        sources.append(
            [
                "Name:=",
                "A1[" + str(idx2) + "," + "1" + "]single1_1_1" + ":1",
                "Magnitude:=",
                str(power) + "W",
                "Phase:=",
                "0deg",
            ]
        )
    return sources


def export_near_field(p, point_name, f):
    oModule3 = oDesign.GetModule("RadField")
    oModule3.ExportFieldsToFile(
        [
            "ExportFileName:=",
            p,
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
    ["Setup1:Sweep"],
    3,
    os.path.join(DIR, DESGIN_NAME + ".s" + str(N) + "p"),
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


for idx in range(0, N):
    oModule1 = oDesign.GetModule("Solutions")
    oModule1.EditSources(edit_sources_array(idx + 1))

    f = 10
    dir_path = os.path.join(DIR, "f" + str(f))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    export_near_field(os.path.join(dir_path, "near_field_rect" + str(idx) + ".nfd"), "rect", f=10)
    export_near_field(os.path.join(dir_path, "near_field_angle" + str(idx) + ".nfd"), "angle", f=10)
    export_near_field(os.path.join(dir_path, "near_field_circle" + str(idx) + ".nfd"), "circle", f=10)

    for f in linspace(8, 12, 101):
        dir_path = os.path.join(DIR, "f" + str(f))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        export_near_field(os.path.join(dir_path, "near_field_single" + str(idx) + ".nfd"), "single", f)
