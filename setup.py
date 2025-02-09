from setuptools import setup, find_packages

setup(
    name='multilingual-speech-emotion-recognition',
    version='0.1.0',
    description='Multilingual Speech Emotion Recognition model...',
    authors=[
        {
            "name": "Luxshan Thavarasa",
            "email": "luxshan.20@cse.mrt.ac.lk",
            "url": "https://www.linkedin.com/in/luxshan-thavarasa",  # Add LinkedIn URL
        },
        {
            "name": "Jubeerathan Thevakumar",
            "email": "jubeerathan.20@cse.mrt.ac.lk",
            "url": "https://www.linkedin.com/in/jubeerathan-thevakumar",  # Add LinkedIn URL
        },
        {
            "name": "Thanikan Sivatheepan",
            "email": "thanikan.20@cse.mrt.ac.lk",
            "url": "https://www.linkedin.com/in/thanikan-sivatheepan",  # Add LinkedIn URL
        },
        {
            "name": "Uthayasanker Thayasivam",
            "email": "uthayasanker.20@cse.mrt.ac.lk",
            "url": "https://www.linkedin.com/in/uthayasanker-thayasivam",  # Add LinkedIn URL
        },
    ],
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