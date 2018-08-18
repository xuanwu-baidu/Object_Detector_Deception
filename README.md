# YOLO_attack_tf

Hi,

This is a demo for attacking YOLO TINY.

How to Use this code:
1. Get into the path of the YOLO_tiny_tf_attack
2. Run command:
    python YOLO_tiny_tf_attack.py -fromfile test/Darren.jpg -frommuskfile test/Darren.xml
3. Tuning the hyperparameter self.punishment and attack steps. Ensure the target condifence is descending below 0.2
4. When meet the end condition, the adversary example will be stored in result
