# RetBul's Retrieval Core - Feature2d Exctraction 

RetBul(Retrieval Building) is an online image retrieval platform applied on a custom dataset of flat buildings in Athens.
RetBul's retrieval mechanism is implemented with feature extraction techniques using the SURF & SIFT image descriptors.
A ranking list of the most to the less relevant to the query image is generated based on the number of inliers measured comparing query and test image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to install the following software.

```
OpenCV 3+
Python 2.7+
NumPy
Pandas
Pyjson-Tricks
```

### Execute

You may notice two versions of the same image descriptor script. 
* The **[descriptor]_single.py** - compares a pair of images.
* The **[descriptor].py** - compares an image to the reference dataset.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
