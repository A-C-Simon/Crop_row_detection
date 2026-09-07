from setuptools import find_packages, setup
import os
from glob import glob

package_name = "mrsim"

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
    maintainer="mrsim",
    maintainer_email="dev@example.com",
    description="MultiROI navigation test rig in Gazebo",
    license="MIT",
    entry_points={
        "console_scripts": [
            "multiroi_nav = mrsim.nav_node:main",
            "multiroi_probe = mrsim.nav_node:probe_main",
        ],
    },
)
