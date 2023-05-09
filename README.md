<div style="display:flex; align-items: center;">
<div style="width:150px">
 <img src="docs/assets/images/logo.png" width="100px">
</div>
<div align="left">
  <h1>Movie Surfer</h1>
</div>
</div>

## Project Description

Movie Surfer is a Python project aimed at developing a movie recommendation system. The goal of this project is to provide users with personalized movie recommendations based on their preferences and viewing history. By analyzing user behavior and movie attributes, the system suggests movies that align with the user's interests, helping them discover new films they might enjoy.

The project leverages various techniques and algorithms commonly used in recommendation systems, such as collaborative filtering, content-based filtering, and hybrid approaches. It utilizes Python as the primary programming language and incorporates popular libraries and tools for data manipulation, analysis, and machine learning.

## Installation

```bash
git clone git@github.com:atsanda/movie-surfer.git
python -m venv env
pip install -r requirements.txt
```

## Poetrt installation:

To install Poetry, follow these steps:

1. Open the command prompt (cmd) and run the following command:
   `curl -sSL https://install.python-poetry.org | python3 - --git https://github.com/python-poetry/poetry.git@master`

2. Check the command line log for the actual environment location. It may have moved due to redirects, links, or junctions. The requested location is "C:\Users\user\AppData\Roaming\pypoetry\venv\Scripts\python.exe" and the actual location is "C:\Users\user\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\Roaming\pypoetry\venv\Scripts\python.exe".

3. Add the actual location to the PATH in System Variables.

4. Restart the command prompt.

5. Verify that Poetry is installed correctly by running the command `poetry --version` in the command prompt.

## Poetry Commands:

Use the following Poetry commands to manage dependencies:

`poetry install`: Install all the dependencies listed in the pyproject.toml file.

`poetry shell`: Activate the virtual environment. If you encounter an error that says "\Scripts\activate.ps1 cannot be loaded because running scripts is disabled on this system", you need to change the execution policy for your Windows OS by running Set-ExecutionPolicy -ExecutionPolicy RemoteSigned in PowerShell (run as administrator).

`poetry add`: Add a new package to the dependency management system. For example, to add the wget package with version 3.2, run the following command:

`poetry add wget==3.2`

`poetry remove`: This command allows you to remove a package from project. Poetry will automatically update the pyproject.toml file and remove the package from the virtual environment.

## Project structure

- [moviesurfer](./moviesurfer)
  The project contains a python package.
  Inside it all the logic is placed, including data processing, metrics, models, etc.
  The code here is supposed to be helpful elsewhere.
- [prototypes](./prototypes)
  contains code for training concrete models using actual data.
- [tests](./tests)
  is used to place all the tests for both the package and the prototypes (if any).
- [data](./data)
  is for local data storage. In this project we use the [MovieLens](https://grouplens.org/datasets/movielens/) dataset.

## Contributing

Contributions to the Movie Surfer project are welcome! If you encounter any issues, have suggestions for new features, or would like to contribute code improvements, please follow these steps:

1. Select an issue and assign it to yourself;
2. Discuss your solution with the core team;
3. Implement the solution in a separate branch, we are using [GitHub Flow](https://user-images.githubusercontent.com/6351798/48032310-63842400-e114-11e8-8db0-06dc0504dcb5.png);
4. Push it and wait for the review;
5. Once all the comments are addressed you are good to go with merging.

You should also install pre-commit hooks locally using `pre-commit install`. More about pre-commit [here](https://pre-commit.com/).

## License

Movie Surfer is licensed under the [MIT License](LICENSE).
