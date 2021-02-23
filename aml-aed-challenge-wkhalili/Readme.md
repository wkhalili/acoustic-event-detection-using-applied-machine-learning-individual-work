The three files were processed the same but building the features were diffirent.


buildign The feature in the tract processing file,first sorting the matrices and then take the percentile,after processing the features (E,s_tract,F_tract) they were not informative,in a way that it is easy for kmeans algorithm to cluster, and to be seperated by SVM.

The same was done in processing tract-Copy2 file , but different extraction of feature ,I extracted features from every feature by spliting the features to 4 pieces every piece is consist of 20 row and then take the percentile after sorting them. but also I have the same result as the original tract features. 

In the PTNE processing, I did the same I sorted the arrays and then took the 95% percentile, 
the data was stretched towards the diagonal and this shape make it difficult for the Kmeans to define the clusters.
I visualized only the features in another file that data is calculated by using both standard deviation and average ,furthermore the visualization let me know before continuing through steps of testing and validation that clustering is imposible with these features.
I mentioned the reason in my report.

From what I read that the SVM fitted a line to seperate the data. but from the visualization plot 
It was (visually) not easy to seperate the data by a line. 

The final report is in the PTNE processing file.
All the work is in the PTNE processing since it was fast to process this type of data.

another thing, for the confusion matrix the first and last row were cut to the half due to matplotlib version regression.I fixed the problem and correct the confusion matrix in PTNE file, but not in the tract files.

Thank you for your time, I appreciate that.





