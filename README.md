# YOLO_attack_tf

Hi,

This is a demo for attacking YOLO TINY.
Paper:
	http://gixnetwork.org/wp-content/uploads/2018/12/Building-Towards-Invisible-Cloak_compressed.pdf

How to Use this code:
1. Get into the path of the YOLO_tiny_tf_attack.

2. Run command:
    python YOLO_tiny_tf_attack.py -fromfile test/Darren.jpg -frommuskfile test/Darren.xml

3. Tuning the hyperparameter self.punishment and attack steps. Ensure the target condifence is below 0.2.

4. When meeting the end condition, the program will save the adversary example will in ./result.

5. See source code for more attack option.


Copyright:

According to the LICENSE file of the original code, Me and original author hold no liability for any 
damages

Do not use this on commercial!

YOLO_tiny Reference from:
	https://github.com/gliese581gg/YOLO_tensorflow

