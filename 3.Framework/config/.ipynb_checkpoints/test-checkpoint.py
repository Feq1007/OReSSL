import argparse  
import config
import micro_cluster_config

args = argparse.ArgumentParser()
mc_cf = micro_cluster_config.get_options(args)
cf = config.get_options(args)



print(cf)
print(mc_cf)