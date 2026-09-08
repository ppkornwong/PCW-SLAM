import os
from glob import glob

from setuptools import find_packages, setup

package_name = "pcwslam"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="p",
    maintainer_email="ppkornwong@gmail.com",
    description="PCW-SLAM: occlusion-aware perception nodes for a LiDAR-inertial SLAM backend.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pcwslam_dynamic_filter     = pcwslam.dynamic_filter:main",
            "pcwslam_underneath         = pcwslam.underneath_detection:main",
            "pcwslam_confidence         = pcwslam.confidence_converter:main",
            "pcwslam_confidence_lmfreeze = pcwslam.confidence_converter_landmark_freeze:main",
            "pcwslam_transmitter        = pcwslam.transmitter_alignment:main",
            "pcwslam_all                = pcwslam.all_in_one:main",
        ],
    },
)
