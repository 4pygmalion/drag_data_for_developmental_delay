# Lightweight deep learning (1D CNN) using drag data for developmental delay


#### Architecutre
Multi-input (multi-modal) deep learning using game touch log.
This logs were extracted from DoBrain for children with developmental delay

![image](https://user-images.githubusercontent.com/45510932/114259359-574cd100-9a08-11eb-9149-bf787d9aed7c.png)


#### Interpretation
We also included interpretation methods (partial grad-cam due to multi-inputs)

figure (A) Grad-CAM result of children with typical development

![image](https://user-images.githubusercontent.com/45510932/114259372-6469c000-9a08-11eb-82dd-d756460ff309.png)

figure (B) Grad-CAM result of children with developmental delay

#### How to use?
```python

# building model
from model import build_1DCNN

TIME_STAMP = 100
N_FEATURES_TS = 12  # Multivariate timeseries
N_FEATRUES_AUX = 25  # Static variables
model = build_1DCNN(TIME_STAMP, N_FEATURES_TS, N_FEATRUES_AUX)


# explanation
from eXplainableAI.model_specific.CNN import GradCAM
g_cam = GradCAM(model, class_index=1, last_conv_name='conv1d_5')  # 'conv1d_5': refer to last conv layer
g_cam.generate_grad_cam()

```

#### Requriment
tensorflow 2.x
seaborn 0.11.x

#### Install
```bash
$ git clone https://github.com/4pygmalion/drag_data_for_developmental_delay
```
