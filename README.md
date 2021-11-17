Name: Catherine Gai

SID: 3034712396

Email: catherine_gai@berkeley.edu

Link to project report website: [file:///Users/catherine_gai/Desktop/Cal/2021%20Fall/CS%20194-26/proj1/deliverables/%20Project%2001.html](deliverables/Project%01.html)

This folder contains four functional python files: "single_scale.py", "multi_scale.py", "utils.py" and "main.py". 



**single_scale.py: **

This python file contains all functions required to run the naive exhaustive approach for aligning channels, as well as automatic contrast and automatic cropping after alignment. 

* single_scale_multi(*params*): exhaustive search with normalized cross correlation metric, used during single iteration in the image pyramid search. 

* single_scale(*params*): exhaustive search with *l2*-norm metric. Used during naive exhaustive search. 

* l2_norm(*params*): calculates the *l2*-norm between two images. 

* ncc(*params*): calculates the normalized cross correlation between two images. 

* generate(*params*): generates synthesized images for all input JPEG low-resolution files. 

* main(): set up directory, collects images, and calls generate. Also called by other programs to run "single_scale.py". 

  

**multi_scale.py: **

This python file contains all functions required to run the image pyramid approach for aligning channels, as well as automatic contrast and automatic cropping after alignment. 

* multi_scale(*params*): executes image pyramid search on given R, G and B channels. Need to specify the number of layers of the pyramid. 
* generate(*params*): generates synthesized images for all input TIFF low-resolution files. 
* main(): set up directory, collects images, and calls generate. Also called by other programs to run pyramid search on required 11 images.
* extra():  set up directory, collects images, and calls generate. Also called by other programs to run pyramid search on additional 2 images.



**utils.py:**

This python file contains all functions for Bells and Whistles part. Two main portions are included here: automatic contrast and automatic cropping. 

* auto_contrast_lab(*params*): provided an RGB image, first transforms it into LAB format and do histogram equalization on the L (luminance) layer. 
* auto_contrast_rgb(*params*): provided an RGB image, do histogram equalization on all three layers respectively. 
* auto_cropping(*params*): provided an RGB image, crop out the boundaries that has only one or two color channels. 



**main.py:**

Run this python file to execute naive exhaustive search or image pyramid method. Three arguments can be specified to control execution of naive exhaustive search (on 3 required JPEG images), image pyramid (on 11 required TIFF images) and extra (2 additional TIFF images). Arguments are specified below: 

* --single_scale (or -s): 
  * Takes on two values "True" and "False" (note that both are string format). 
  * Default value: "True".
  * If "True", runs the naive exhaustive search on 3 required JPEG images. 
* --multi_scale (or -m):
  * Takes on two values "True" and "False" (note that both are string format). 
  * Default value: "True"
  * If "True", runs the image pyramid method on 11 required TIFF images. 
* --extra (or -e):
  * Takes on two values "True" and "False" (not that both are string format). 
  * Default value: "True"
  * If "True", runs the image pyramid method on 2 additional TIFF images. 

Sample execution commands: 

* `python main.py -s="True" -m="False" -e="False"`: only performs exhaustive search on 3 required JPEG images. 
* `python main.py -m="False"`: performs exhaustive search on 3 required JPEG images and image pyramid method on 2 additional TIFF images. 
* `python main.py`: performs all executions. 



