import matplotlib.colors as mcolors

INPUT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/hic'
# OUTPUT_DIR = '/home/hc0783@unt.ad.unt.edu/workspace/hicinterpolate/datasets/timeseries/triplets'
PATCH_OVERLAP_RATIO = 0.2

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "juicebox", ["#FFFFFF", "#FFAAAA", "#FF5555", "#FF0000", "#B30000"], N=256
)

RESOLUTIONS = [25000, 10000, 5000]
PATCH_SIZES = [64]
STEP_SIZES = [400, 200, 100, 50]


# TRAIN_CHROMOSOMES = {
#     "human": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
#               "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "X", "Y"]
#     # "mouse": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
#     #           "13", "14", "15", "16", "17", "18", "19", "X", "Y"]
# }

CHROMOSOMES = {
    "human": ["10", "11", "15", "16", "20", "21"],
    "mouse": ["10", "11", "15", "16", "18", "19"]
}

# CHROMOSOMES = {
#     "human": ["21"],
#     "mouse": ["19"]
# }

DATASET = {
    "human": {
        "dmso": {
            "control": {
                "triplets":
                [
                    ["dmso_control_0m",
                     "dmso_control_30m",
                     "dmso_control_60m"],

                    ["dmso_control_30m",
                     "dmso_control_60m",
                     "dmso_control_90m"],

                    ["dmso_control_60m",
                     "dmso_control_90m",
                     "dmso_control_120m"]
                ]
            }
        },
        # Train
        "dtag": {
            "v1": {
                "triplets":
                [
                    ["dtag_v1_0m",
                     "dtag_v1_30m",
                     "dtag_v1_60m"],

                    ["dtag_v1_30m",
                     "dtag_v1_60m",
                     "dtag_v1_90m"],

                    ["dtag_v1_60m",
                     "dtag_v1_90m",
                     "dtag_v1_120m"]
                ]
            }
        },
        # Train
        # "k562": {
        #     "dpnii": {
        #         "triplets":
        #         [
        #             ["k562_dpnii_60m",
        #              "k562_dpnii_120m",
        #              "k562_dpnii_180m"],

        #             ["k562_dpnii_120m",
        #              "k562_dpnii_180m",
        #              "k562_dpnii_240m"],

        #             ["k562_dpnii_480m",
        #              "k562_dpnii_960m",
        #              "k562_dpnii_1440m"]
        #         ]
        #     },
        #     "buffer": {
        #         "triplets":
        #         [
        #             ["k562_buffer_480m",
        #              "k562_buffer_960m",
        #              "k562_buffer_1440m"]
        #         ]
        #     }
        # },
        # "gm12878": {
        #     "rt": {
        #         "triplets": [
        #             ["gm12878_rt_1m",
        #              "gm12878_rt_5m",
        #              "gm12878_rt_10m"]
        #         ]
        #     },
        #     "37c": {
        #         "triplets": [
        #             ["gm12878_37c_1m",
        #              "gm12878_37c_5m",
        #              "gm12878_37c_10m"]
        #         ]
        #     }
        # },
        # Test
        "wtc11": {
            "atrial": {
                "triplets":
                [
                    ["wtc11_atrial_2880m",
                     "wtc11_atrial_5760m",
                     "wtc11_atrial_8640m"]
                ]
            },
            "ventricular": {
                "triplets":
                [
                    ["wtc11_ventricular_2880m",
                     "wtc11_ventricular_5760m",
                     "wtc11_ventricular_8640m"]
                ]
            }

        }
    },
    "mouse": {
        # Test
        "cerebellar_granule_neuron": {
            "control": {
                "triplets":
                [
                    ["cerebellar_granule_neuron_control_10080m",
                     "cerebellar_granule_neuron_control_11520m",
                     "cerebellar_granule_neuron_control_12960m"]
                ]
            }
        },
        # Test
        "embryo": {
            "development": {
                "triplets": [
                    ["sperm",
                     "mii_oocyte",
                     "zygote"],

                    ["mii_oocyte",
                     "zygote",
                     "early2_cell"],

                    ["zygote",
                     "early2_cell",
                     "late2_cell"],

                    ["early2_cell",
                     "late2_cell",
                     "8cell"],

                    ["late2_cell",
                     "8cell",
                     "icm"],

                    ["8cell",
                     "icm",
                     "mes_cell"]
                ]
            }
        }
    }
}
