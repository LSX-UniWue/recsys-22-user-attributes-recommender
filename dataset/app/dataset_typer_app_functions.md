## Dataset CLI ##
The pre-processing of data set sin this project is done with the help of a CLI which provides the following
functions:
0. (Downloading of data sets Not available for all datasets)
1. Pre-processing of data 
2. Indexing of data 
3. Splitting of the data set into train, test and validation split


The structure of the CLI is as follows: 
* dataset_app.py
    * \<index\> 
    * \<pre_process\>
    * \<split\>   
        * \<next_item\>
        * \<ratio\>
    * \<vocabulary\>