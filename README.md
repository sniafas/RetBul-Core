# RetBul's Image Retrieval Core - Feature2d Extraction

RetBul(Retrieval Building) is an online image retrieval platform applied on a custom dataset of flat buildings in Athens.
RetBul's retrieval mechanism is implemented with feature extraction techniques using the SURF & SIFT image descriptors.
A ranking list of the most to the less relevant to the query image is generated based on the number of inliers measured comparing query and test image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to install the following software.

* OpenCV 3+
* Python 2.7+
* NumPy
* Pandas
* Pyjson-Tricks
* [VyronasDb](https://retbul.sniafas.eu/download) - Vyronas Database

## Extraction

Extract handcrafted image features, build a ranking list with the number of inliers, of the most to the less relevant to the query image.

* **[descriptor]_single.py** - 1 to 1 query images.
* **[descriptor].py** - many to many query images. Setup one or more building queries to the reference dataset. 

## Preprocessing

Parsing the extracted data in order to evaluate them in:
* Precision, Recall, F-Measure
* Eleven Point Interpolation
* Highest F-Measure / Building
* Mean Average Precision

## Evaluation

Plotting the processed data


**Extraction and preprocessing would run only in a Linux machine.**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
