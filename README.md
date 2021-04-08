# Lightweight deep learning (1D CNN) using drag data for developmental delay


##### Architecutre
Multi-input (multi-modal) deep learning using game touch log.
This logs were extracted from DoBrain for children with developmental delay


##### Interpretation
We also included interpretation methods (partial grad-cam due to multi-inputs)


##### usage
```python

# explanation
from eXplainableAI.model_specific.CNN import GradCAM
g_cam = GradCAM(model, class_index=1, last_conv_name='conv1d_5')  # 'conv1d_5': refer to last conv layer
g_cam.generate_grad_cam()

```

##### Requriment
tensorflow 2.x
seaborn 0.11.x
