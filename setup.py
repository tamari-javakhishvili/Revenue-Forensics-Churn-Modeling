from setuptools import setup, find_packages

setup(
    name="customer_churn_forensics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "pytest"
    ],
)