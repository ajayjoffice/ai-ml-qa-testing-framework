# Dataset source

`raw/Customer Churn.csv` is the Iranian Churn dataset from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- Citation: *Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository.*
- DOI: https://doi.org/10.24432/C5JW3Z
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Contents: 3,150 records, 13 input fields, and the binary `Churn` target.

The CSV is kept in the project so setup and CI can run without downloading data at test time. The source includes no customer identifier. It has repeated full feature profiles, so the training loader validates required columns and value ranges but does not reject repeated profiles as duplicate customers.

The dataset is historical sample data and is used only for this QA/testing demonstration. It is not suitable as a current production churn model without further review and validation.
