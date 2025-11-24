- Overview
Short description of what NAS is and why hardware-aware optimization matters.
https://project.autocoder.cc/PROJ_dac37f19/admindashboardpage
- Features
- Real-time latency measurement
- Energy consumption profiling
- Quantization-aware architecture search
- Deployment on smartphones/embedded systems
- Tech Stack
TensorFlow Lite, PyTorch Mobile, ONNX Runtime, AutoKeras/NNI

- Installation
git clone https://github.com/yourusername/hardware-aware-nas.git
cd hardware-aware-nas
pip install -r requirements.txt

- Usage
python src/nas/run_search.py --config configs/mobile_search.yaml
python src/deployment/export_tflite.py --model best_model.pth

- Results
Include plots of latency vs accuracy, energy profiling charts, screenshots of deployment on mobile.
- Applications
Object detection, face recognition on smartphones/IoT devices.
- Contributing & License
Encourage collaboration, specify license (MIT, Apache 2.0, etc.).

Related works on efficient deep learning
MicroNet for Efficient Language Modeling

Lite Transformer with Long-Short Range Attention

AMC: AutoML for Model Compression and Acceleration on Mobile Devices

Once-for-All: Train One Network and Specialize it for Efficient Deployment

ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware

