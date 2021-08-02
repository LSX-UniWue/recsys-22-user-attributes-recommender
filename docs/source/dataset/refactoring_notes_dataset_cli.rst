Dataset Directory
-----------------

movielens.py:
~~~~~~~~~~~~~

-  preprocess\_data

   -  Moved to dataset\_preprocessing/movielens.py

-  build vocabularies

   -  Moved to dataset\_preprocessing/utils.py

-  read\_csv

   -  Moved to dataset\_preprocessing/utils.py

-  \_get\_position\_with\_offset

   -  Moved to datset\_splits/conditional\_split.py

-  split\_dataset

   -  moved into dataset\_preprocessing/movielens.py

-  main

   -  Reworked as movielens command in
      dataset/app/commands/data\_set\_commands.py
   -  And as download\_and\_unzip\_movielens\_data() in
      dataset\_preprocessing/movielens.py

quick\_indices\_and\_splits.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Unchanged

utils.py (Deleted)
~~~~~~~~~~~~~~~~~~

-  download\_dataset

   -  Moved to dataset\_preprocessing/utils.py

-  unzip\_file

   -  Moved to dataset\_preprocessing/utils.py

Runner Directory
----------------

stats/build\_populatity\_stats.py (Deleted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  create\_conditional\_index moved to
   dataset/popularity/build\_popularity.py and renamed as build

create\_conditional\_index.py (Deleted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moved into split commands.py:

-  create\_conditional\_index and dataset/dataset\_splits/conditional\_index.py:
-  filter\_by\_sequence\_feature
-  \_build\_target\_position\_extractor
-  create\_conditional\_index\_using\_extractor

create\_csv\_dataset\_splits.py (Deleted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Has been reworked as ratios command under
   dataset/app/split\_commands.py
-  Code moved to dataset/dataset\_splits/ratio\_split.py

create\_reader\_index (Deleted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Code moved to dataset/app/index\_command.py
-  create\_index\_for\_csv renamed to index\_csv

