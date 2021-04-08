import numpy as np
import tensorflow as tf
import cv2

# GRAD CAM

class GradCAM(object):

    def __init__(self, model, class_index, last_conv_name):
        self.model = model
        self.class_index = class_index
        self.layer_name = last_conv_name

    def _create_sub_graph(self):
        '''
        Parameters
        ----------
        model: tf.keras.model.Model
        layer_name: str

        Return
        ------
        sub graph: model: tf.keras.model.Model with multiple output
            it's outputs are (last conv output, model final output(score))
        '''

        subgraph = tf.keras.models.Model([self.model.inputs],
                                         [self.model.get_layer(self.layer_name).output, self.model.output])

        return subgraph

    def generate_grad_cam(self,
                          img_tensor,
                          alpha=0.3,
                          cmap=cv2.COLORMAP_VIRIDIS):
        """
        params:
        -------
        img_tensor: Array-like. 2D image. shape=(X, Y, channels)
        model: tensorflow.keras.mode
        class_index: output class
        activation_layer: the name of last convolutional layer

        return
        -------
        grad_cam: np.array with overlay pixel wised importance
        """
        X_ = np.array(img_tensor, dtype='float32')
        y_c = self.model.output.op.inputs[0][0, self.class_index]
        A_k = self.model.get_layer(self.layer_name).output

        # Compute gradient
        with tf.GradientTape() as tape:
            sub_graph = self._create_sub_graph()
            sub_graph.layers[-1].activation = tf.keras.activations.relu  # Activation modified
            conv_outputs, prediction = sub_graph(np.array([X_]))
            loss = prediction[:, self.class_index]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam.numpy(), (128, 128))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255 * heatmap),
                                colormap=cmap)

        output_image = cv2.addWeighted(cv2.cvtColor(img_tensor.astype('uint8'), cv2.COLOR_RGB2BGR),
                                       alpha=alpha,
                                       src2=cam,
                                       beta=1,
                                       gamma=0)

        return output_image