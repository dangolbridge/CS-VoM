"""Central mesh-region and connector-tag definitions for the CS/VoM workflow.

Keep anatomical labels and generated connector labels in one place so scripts
never need to duplicate numeric constants.  The surface-region values below are
the standard AtrialMTK labels used by this workflow; fibrosis-specific labels
(100/101/102/200/201/202) are deliberately not included.
"""

# Standard AtrialMTK surface/structure regions used by this project.
LA_EPI_REGIONS = (12, 14, 22, 24, 26, 28)
LA_ENDO_REGIONS = (11, 13, 21, 23, 25, 27)
RA_EPI_REGIONS = (1, 2, 5, 6, 7)

SAN_REGION = 3
CT_REGION = 8
PM_REGION = 9
BB_REGION = 10

# Line-element tags added by build_biatrial_connections.py.
LA_BILAYER_CONNECTION_TAG = 32
SAN_CONNECTION_TAG = 33
CT_CONNECTION_TAG = 34
PM_CONNECTION_TAG = 35
FO_CONNECTION_TAG = 39
CS_LA_CONNECTION_TAG = 40
BB_ATRIAL_CONNECTION_TAG = 41
VOM_LA_CONNECTION_TAG = 50

# Surface tags added by build_bb_bridges.py.
BB_CLOSEST_BRIDGE_TAG = 70
BB_ANATOMICAL_BRIDGE_TAG = 80

# Backward-compatible aliases used internally by some scripts.
TAG_LA_BILAYER = LA_BILAYER_CONNECTION_TAG
TAG_SAN = SAN_CONNECTION_TAG
TAG_CT = CT_CONNECTION_TAG
TAG_PM = PM_CONNECTION_TAG
TAG_FO = FO_CONNECTION_TAG
TAG_BB = BB_ATRIAL_CONNECTION_TAG
