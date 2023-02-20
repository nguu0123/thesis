from setuptools import find_packages, setup

setup(
    name='qoa4ml',
    version='0.0.62',
    description='Quality of Analysis for Machine Learning',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type='text/markdown',
    url='https://rdsea.github.io/',
    author='Aaltosea',
    email='tri.m.nguyen@aalto.fi',
    keyword='Monitoring Machine Learning',
    install_requires=['pika','requests','paho-mqtt','prometheus-client','psutil', 'docker'],
    packages=find_packages(),
    license='Apache License 2.0'
)