from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.network_path = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/anti-UAV/Anti-UAV-master/Eval/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh/otb/PYTHON2/tracker_benchmark/data'
    settings.results_path = '/media/hp/01a64147-0526-48e6-803a-383ca12a7cad/WH/wh2020/anti-UAV/Anti-UAV-master/Eval/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''

    return settings

