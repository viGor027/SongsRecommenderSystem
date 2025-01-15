from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "torch",
    "pytorch-lightning",
    "lightning",
    "google-cloud-storage",
]

setup(
    name="SongsRecommenderSystem",
    version="0.1.0",
    description="System rekomendacji piosenek",
    author="Twoje Imię",
    author_email="twojemail@example.com",
    packages=find_packages(include=["cloud", "cloud.*",
                                    "model_components", "model_components.*",
                                    "prototyping", "prototyping.*", "song_pipeline.constants"]),
    package_data={
        "cloud": ["key.json"],
    },
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    setup_requires=REQUIRED_PACKAGES,
    python_requires=">=3.8",
)
