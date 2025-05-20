
# Image Pre process

Before classification lets apply few filters on the image so that any noise is reduced and edges are emphasised. This code helps apply the preprocessing steps for all the images in a dataset



Download a dataset make sure the naming in the folder is sequential according to the code. Place both the dataset folder and this python code in same directory. The outputs are saved into a "output" directory.

# Example Folder structure
Project\
-crack\
--crack1.jpg\
--crack2.jpg\
--.\
--.\
--.\
-main.py
 
To run the code pass "crack" as an argument to the function in the main code.

# Sequential Fitering
But to train a classifier it would be easy to just have single output. Hence the sequential filtering
One more code file "test.py" takes the images from the dataset and applies these filters one upon one and gives a singular output. That singular output gives segmented image used for classification
