import setuptools

setuptools.setup(
    name="single-peaked-bandits",
    author="David Lindner, et al.",
    description="Addressing the Long-term Impact of ML-based Decisions via Policy Regret",
    version="0.1dev",
    long_description=open("README.md").read(),
    install_requires=[
        "numpy",
        "matplotlib",
        "seaborn",
        "pulp",
    ],
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
)
