from setuptools import setup, find_packages

setup(
    name='multilingual-speech-emotion-recognition',
    version='0.1.0',
    description='Multilingual Speech Emotion Recognition model...',
    author='Luxshan Thavarasa, Jubeerathan Thevakumar, Thanikan Sivatheepan, Uthayasanker Thayasivam',
    author_email='luxshan.20@cse.mrt.ac.lk, jubeerathan.20@cse.mrt.ac.lk, thanikan.20@cse.mrt.ac.lk, uthayasanker.20@cse.mrt.ac.lk',
    license='Apache-2.0',
    packages=find_packages(exclude=['tests*']),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.6.0,<3.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
