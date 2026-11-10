from setuptools import find_packages, setup

package_name = "traffic_light_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="solidSnakesado",
    maintainer_email="sado14th@gmail.com",
    description="Traffic light perception, smoothing and control nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "env_check = traffic_light_nodes.env_check:main",
            "video_source = traffic_light_nodes.video_source:main",
            "perception_node = traffic_light_nodes.perception_node:main",
            "raw_recorder = traffic_light_nodes.raw_recorder:main",
        ],
    },
)
