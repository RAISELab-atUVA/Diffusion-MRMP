from setuptools import setup, find_packages
from codecs import open
from os import path


from smd import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='smd',
      version=__version__,
      description='Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models',
      author='Jinhao Liang',
      author_email='jliang@email.virginia.edu',
      packages=find_packages(where=''),
      install_requires=requires_list,
      )
