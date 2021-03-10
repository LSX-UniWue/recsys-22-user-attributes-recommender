# Indices created in this project

In this project for every session data set file three indices are created: session indices, next item indices, and 
leave one out indices.

### Session Indices ###
Includes the start and end of every session in a data set. It is implemented in the class CsvSessionIndexer in 
[indexer.py](../../data/base/indexer.py).

### Next Item Indices ###
Includes entries for the start and end of every session as well as the start and end of every sub-sequence of a session.
The next item index also includes a target item for every entry. The index is implemented using the SessionPositionIndex
in the SessionPositionIndexBuilder class in [index_builder.py](../../data/datasets/index_builder.py).

E.g. for session <br>
0 -> 1 -> 2 -> 3<br>
0 -> [1]<br>
0 -> 1 -> [2]<br>
0 -> 1 -> 2 -> [3]

where [] denotes the target item

### Leave One Out Indices
Includes entries for the start and end of every session as well as a target item for every session.
The index is implemented using the SessionPositionIndex in the SessionPositionIndexBuilder class in 
[index_builder.py](../../data/datasets/index_builder.py).

E.g. for session <br>
0 -> 1 -> 2 -> 3<br>
0 -> 1 -> 2 -> [3]

where `[]` denotes the target item

### Data Set File Scheme
In order to use both Leave one out and Next Item recommendation tasks for training on a data set the following files have
to be stored/created under \<directory-basepath\>. 

\<prefix\>.csv - The original data file containing the raw data.

\<prefix\>.session.idx - The session index for the original raw data. 

\<prefix\>.train.csv - data file containing share of session that are used for training with the next item 
recommendation task

\<prefix\>.train.session.idx - Session index for \<prefix\>.train.csv

\<prefix\>.train.nextitem.idx - Next Item Index for \<prefix\>.train.csv

\<prefix\>.validation.csv - data file containing share of session that are used for validation with the next item 
recommendation task

\<prefix\>.validation.session.idx - Session index for \<prefix\>.validation.csv

\<prefix\>.validation.nextitem.idx - Next Item Index for \<prefix\>.validation.csv

\<prefix\>.test.csv - data file containing share of session that are used for testing with the next item 
recommendation task

\<prefix\>.test.session.idx - Session index for \<prefix\>.test.csv

\<prefix\>.test.nextitem.idx - Next Item Index for \<prefix\>.test.csv

\<prefix\>.validation.loo.idx - Leave One Out Index for \<prefix\>.csv used for training and validation with the leave
one out trainings task

\<prefix\>.test.loo.idx - Leave One Out Index for \<prefix\>.csv used for testing with the leave one out trainings task

\<prefix\>.vocabulary.\<columnname\>.txt - List of unique entries of a column

\<prefix\>.popularity.\<columnname\>.txt - Percentage of occurrences of an item
