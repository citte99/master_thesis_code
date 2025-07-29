from setuptools import setup, find_packages
from setuptools import setup, find_packages

# setup(
#     name="my_package",
#     version="0.1.0",
#     package_dir={"": "src"},
#     packages=find_packages(where="src"),
# )


setup(
  name="master_thesis_code",
  version="0.1.0",
  packages=find_packages(where="src"),     # <-- include every package under src/
  package_dir={"": "src"},                # <-- map the root to src/
  install_requires=[
      'wandb',
      'astropy',
      'shapely',
      'h5py',
      'scikit-learn',
      'numba'
 #     'torch'
  ],
  entry_points = { # these I do not need for now
      # 'console_scripts' : [
      #     'train = my_package.train:main'
      # ]
  }
)