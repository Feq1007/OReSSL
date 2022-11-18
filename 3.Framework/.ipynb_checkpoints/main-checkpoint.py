import argparse
import config.config as cfg

opt = cfg.get_options(argparse.ArgumentParser())
print(opt)