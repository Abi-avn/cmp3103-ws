from setuptools import find_packages, setup

package_name = 'temp_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_3d = temp_1.detector_3d:main',
            'tidybot_complexity1 = temp_1.tidybot_complexity1:main',
            'tidybot_complexity2 = temp_1.tidybot_complexity2:main',
            'tidybot_irl = temp_1.tidybot_irl:main',
            'patch_mapper = temp_1.patch_mapper:main',
            'go_location = temp_1.go_location:main',

        ],
    },
)
