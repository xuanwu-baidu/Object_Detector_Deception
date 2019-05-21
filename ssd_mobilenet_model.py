import tensorflow as tf
import numpy as np
import cv2

import sys
import argparse
import pdb

"""# Ensemble of models"""
NAME_TO_MODEL = {
    'rcnn_inceptionv2': {'path': 'parameters/faster_rcnn_inception_v2_coco_new',
                    'input_tensor': 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0',
                    'logits_tensor': 'SecondStagePostprocessor/scale_logits:0',
                    'box_tensor': 'SecondStagePostprocessor/Reshape_4:0',
                    'score_tensor': 'SecondStagePostprocessor/convert_scores:0',
                    'feature_tensor_1': 'Conv/Relu6:0',
                    'feature_tensor_2': 'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Conv2d_2b_1x1/Relu:0',
                    'target_size': 600,
                    'type':'rcnn'
    },
    'rcnn_resnet101': {'path': 'parameters/faster_rcnn_resnet101_new',
                    'input_tensor': 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0',
                    'logits_tensor': 'SecondStagePostprocessor/scale_logits:0',
                    'box_tensor': 'SecondStagePostprocessor/Reshape_4:0',
                    'score_tensor': 'SecondStagePostprocessor/convert_scores:0',
                    'feature_tensor_1': 'Conv/Relu6:0',
                    'feature_tensor_2': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/conv1/Relu:0',
                    'target_size': 600,
                    'type':'rcnn'
    },
    'rcnn_resnet50': {'path': 'parameters/faster_rcnn_resnet50_coco_2018_01_28',
                    'input_tensor': 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0',
                    'logits_tensor': 'SecondStagePostprocessor/scale_logits:0',
                    'box_tensor': 'SecondStagePostprocessor/Reshape_4:0',
                    'score_tensor': 'SecondStagePostprocessor/convert_scores:0',
                    'feature_tensor_1': 'Conv/Relu6:0',
                    'feature_tensor_2': 'FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/conv1/Relu:0',
                    'target_size': 600,
                    'type':'rcnn'
    },
    'rfcn_resnet101': {'path': 'parameters/rfcn_resnet101_coco_2018_01_28',
                    'input_tensor': 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0',
                    'logits_tensor': 'SecondStagePostprocessor/scale_logits:0',
                    'box_tensor': 'SecondStagePostprocessor/Reshape_4:0',
                    'score_tensor': 'SecondStagePostprocessor/convert_scores:0',
                    'feature_tensor_1': 'Conv/Relu6:0',
                    'feature_tensor_2': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/conv1/Relu:0',
                    'target_size': 600,
                    'type':'rfcn'
    },
    'ssd_inceptionv2': {'path': 'parameters/ssd_inception_v2_coco_new',
                    'input_tensor': 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0',
                    'logits_tensor': 'Postprocessor/scale_logits:0',
                    'box_tensor': 'Postprocessor/Reshape_2:0',
                    'score_tensor': 'Postprocessor/convert_scores:0',
                    'feature_tensor_1': 'FeatureExtractor/InceptionV2/InceptionV2/Conv2d_2c_3x3/Relu6:0',
                    'feature_tensor_2': 'FeatureExtractor/InceptionV2/InceptionV2/Conv2d_2b_1x1/Relu6:0',
                    'target_size': 300,
                    'type':'ssd'
    },
    'ssd_mobilenetv1': {'path': './weights/ssd_mobilenet_v1_coco_new',
                    'input_tensor': 'ToFloat:0',
                    'logits_tensor': 'Postprocessor/scale_logits:0',
                    'box_tensor': 'Postprocessor/Reshape_2:0',
                    'score_tensor': 'Postprocessor/convert_scores:0',
                    'target_size': 300,
                    'type':'ssd'
    },
    'ssd_mobilenetv2': {'path': 'parameters/ssd_mobilenet_v2_coco_2018_03_29',
                    'input_tensor': 'ToFloat:0',
                    'logits_tensor': 'Postprocessor/scale_logits:0',
                    'box_tensor': 'Postprocessor/Reshape_2:0',
                    'score_tensor': 'Postprocessor/convert_scores:0',
                    'target_size': 300,
                    'type':'ssd'
    },
    'ssd_resnet50': {'path': 'parameters/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
                    'input_tensor': 'ToFloat:0',
                    'logits_tensor': 'Postprocessor/scale_logits:0',
                    'box_tensor': 'Postprocessor/Reshape_2:0',
                    'score_tensor': 'Postprocessor/convert_scores:0',
                    'target_size': 640,
                    'type':'ssd'
    },
}

model_name = 'ssd_mobilenetv1'

def build_graph(scope_name):
    detection_graph = tf.train.import_meta_graph(model_args['path'] + '/model.ckpt' + '.meta', 
                                                         import_scope=scope_name,
                                                         clear_devices=True)  # ,input_map=graph_map)

    detection_graph.restore(sess, model_args['path'] + '/model.ckpt')

    input_tensor = graph.get_tensor_by_name(graph_name + model_args['input_tensor'])
    second_stage_cls_scores_ = graph.get_tensor_by_name(graph_name + model_args['score_tensor'])
    second_stage_cls_logits_ = graph.get_tensor_by_name(graph_name + model_args['logits_tensor'])

def main(args):
    # init graph
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    
    model_args = NAME_TO_MODEL[model_name]
    
    graph = tf.Graph()
    sess = tf.Session(graph=graph, config=tfconfig)
    
    # get img
    test_img = cv2.imread("./test/person.jpg")    # ("bycle.png")
    test_img = cv2.resize(test_img, (model_args['target_size'],model_args['target_size']))
    # test_img = (test_img/255.0)*2.0-1.0
    
    with graph.as_default():
        # image_input_ = tf.placeholder(tf.float32, shape=(None, 300, 300, 3), name='image_input')

        # create model graph input map
        # graph_map = {model_args['input_tensor']:patched_input_}
        graph_name = 'detection/'
        #ckpt = tf.train.get_checkpoint_state(model['path'])
        with sess:
            
            detection_graph = tf.train.import_meta_graph(model_args['path'] + '/model.ckpt' + '.meta', 
                                                         import_scope='detection',
                                                         clear_devices=True)  # ,input_map=graph_map)

            detection_graph.restore(sess, model_args['path'] + '/model.ckpt')

            input_tensor = graph.get_tensor_by_name(graph_name + model_args['input_tensor'])
            second_stage_cls_scores_ = graph.get_tensor_by_name(graph_name + model_args['score_tensor'])
            second_stage_cls_logits_ = graph.get_tensor_by_name(graph_name + model_args['logits_tensor'])

            print("Successfully Loaded!")
            v1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            
            # test 
            input_batch = test_img[np.newaxis, :]
            
            feed_dict = {input_tensor: input_batch}
            fetch_list = [second_stage_cls_scores_, second_stage_cls_logits_]
            
            result = sess.run(fetch_list, feed_dict)
            
            print(result[0][0,:,1].max(), result[1][0,:,1].max())
            pdb.set_trace()
            
            print(result[1].shape)            
            
            for i in range(1):
                detection_graph = tf.train.import_meta_graph(model_args['path'] + '/model.ckpt' + '.meta', 
                                                             import_scope='detection'+str(i),
                                                             clear_devices=True)  # ,input_map=graph_map)
            
            detection_graph.restore(sess, model_args['path'] + '/model.ckpt')
            graph_name0 = 'detection0/'
            
            input_tensor0 = graph.get_tensor_by_name(graph_name0 + model_args['input_tensor'])
            second_stage_cls_scores_0 = graph.get_tensor_by_name(graph_name0 + model_args['score_tensor'])
            second_stage_cls_logits_0 = graph.get_tensor_by_name(graph_name0 + model_args['logits_tensor'])

            print("Successfully Loaded!")
            v2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            
            feed_dict0 = {input_tensor0: input_batch}
            fetch_list0 = [second_stage_cls_scores_0, second_stage_cls_logits_0]
            
            result0 = sess.run(fetch_list0, feed_dict0)
            
            print(result0[0][0,:,1].max(), result0[1][0,:,1].max())
            pdb.set_trace()
            
            print(result0[1].shape)     
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_folder_dir', type=str, help='Dir with videos to be processed.')
    parser.add_argument('output_folder_dir', type=str, help='Dir for processed videos to save.')
    parser.add_argument('method', type=str, help='Processing method for each frame.')
    
    parser.add_argument('--useEOT', type=str,
                        help='Create EOT attack graph instead of single angle graph.', default=44)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
