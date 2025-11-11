# MASKvla

pip install -r requirements.txt
### install libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

<!-- pip install -r experiments/robot/libero/libero_requirements.txt -->
### install clip

git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .

### download ckpt
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download VQ-VLA/vq-vla-weight --local-dir ckpt_vqvla 