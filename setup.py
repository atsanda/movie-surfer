from setuptools import setup

setup(
    name="moviesurfer",
    packages=["moviesurfer"],
    version="0.1.0",
    description=("Core library for protyping movie recommendation systems"),
    author="Artyom Tsanda, Anna Kiseleva, Islam Yangurazov",
    license="MIT",
    entry_points={
        "console_scripts": ["moviesurfer=moviesurfer.cli.main:cli"],
    },
)
