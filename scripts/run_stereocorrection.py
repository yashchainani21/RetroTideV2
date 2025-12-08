"""
"""
# pylint: disable=no-member, import-error, wrong-import-position
from typing import Optional, List
import yaml
import bcs

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
config = load_config("RetroTideV2/krswaps/test_set/DEBS.yaml")

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """
    Modifies the starter and extender units available for RetroTide.
    Removes all starter and extender units not specifed in the input lists.
    """
    for key in list(bcs.starters.keys()):
        if key not in starter_codes:
            bcs.starters.pop(key, None)
    for key in list(bcs.extenders.keys()):
        if key not in extender_codes:
            bcs.extenders.pop(key, None)

modify_bcs_starters_extenders(starter_codes = config["starter_codes"],
                              extender_codes = config["extender_codes"])

from krswaps import krswaps as krs

results = krs.krswaps_stereo_correction(config['molecule'], config['stereo'], config['offload_mech'])
krs.output_results(results, config['job_name'], config['output_dir'])
        