# Decision Tree 🌳

Decision Tree is a program that constructs, validates, and tests a decision tree given the training, validation, and testing data. These decision trees are constructed either using information gain or variance impurity metrics. Additionally, the program has an option to print out the tree to see the specific features each tree found to be most important. This repository contains two different datasets you can test the model with or you can create your own dataset and try running the program through it!

## Getting Started

These instructions will give you a copy of the neural network up and running on
your local machine for development and testing purposes.

### Prerequisites

To run this application locally on your computer, you'll need `Git` and `python3` installed on your computer.

### Installing

Then run the following command in the command line and go to the desired directory to store this project:

Clone this repository:

    git clone https://github.com/JonathanCen/Decision-Tree.git

Navigate to the cloned repository:

    cd Decision-Tree

To run the program on the data_set_1 and split using information gain:

    python3 Decision_Tree.py ./data_set_1/training_set.csv ./data_set_1/validation_set.csv ./data_set_1/test_set.csv no h1

## More on program arguments

The format to run the program is:

    python3 Decision_Tree.py training_file_path validation_file_path test_file_path print_tree metric

- training_file_path: the file path to the training data
- validation_file_path: the file path to the validation data
- test_file_path: the file path to the test data
- print_tree: 'yes' or 'no' to printing the tree
- metric: 'h1' for information gain and 'h2' for variance impurity

## Contributing

All issues and feature requests are welcome.
Feel free to check the [issues page](https://github.com/JonathanCen/Decision-Tree/issues) if you want to contribute.

## Authors

- **Jonathan Cen** - [LinkedIn](https://www.linkedin.com/in/jonathancen/), [Github](https://github.com/JonathanCen)

## License

Copyright © 2022 [Jonathan Cen](<ADD PERSONAL WEBSITE LINK>).\
This project is [MIT licensed](https://github.com/JonathanCen/Decision-Tree/blob/main/LICENSE).
