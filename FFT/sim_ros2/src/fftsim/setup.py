from setuptools import find_packages, setup
import os
from glob import glob

package_name = "fftsim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world") + glob("worlds/*.spawn.json")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.urdf")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fftsim",
    maintainer_email="dev@example.com",
    description="FFT crop-row navigation test rig in Gazebo",
    license="MIT",
    entry_points={
        "console_scripts": [
            "fft_nav = fftsim.nav_node:main",
            "fft_probe = fftsim.nav_node:probe_main",
        ],
    },
)
