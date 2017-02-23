import os
import subprocess



from setuptools import setup, find_packages

subprocess.call(['make','-C','sixtracklib'])

setup(
        name='sixtracklib',
        version='0.0.0',
        description='6D Tracking Library',
        author='Riccardo De Maria',
        author_email='riccardo.de.maria@cern.ch',
        url='https://github.com/rdemaria/sixtracklib',
        packages=find_packages(),
        package_dir={'sixtracklib': 'sixtracklib'},
        install_requires=['numpy'],
        package_data={'sixtracklib': ['block.so']},
        include_package_data=True
)
