import tensorflow as tf
import pdb

# build graph
init = tf.constant_initializer([23,42])
var2 = tf.get_variable(name="var2",shape=[1,2],dtype=tf.float32,initializer=init)

initializer=tf.constant_initializer(value=3)
var3 = tf.get_variable(name="var3",shape=[1],dtype=tf.float32,initializer=initializer)

tf.get_variable_scope().reuse_variables()
var_list = tf.contrib.framework.get_variables()
var3_reuse = tf.get_variable(name=var_list[1].name[:-2])
pdb.set_trace()



sess = tf.Session()
sess.run(tf.global_variables_initializer())
'''
in_dict = {self.x: inputs,
           self.punishment:punishment}
'''
result = sess.run([var2,var3,var3_reuse])#,feed_dict=in_dict
pdb.set_trace()
print(result[0])