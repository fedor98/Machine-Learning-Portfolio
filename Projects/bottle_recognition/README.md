0.
install (mi)niconda if you haven't: https://docs.anaconda.com/miniconda/
and create an venv with python 3.10
conda create -n <env_name> python=3.10

1.
pip install -r requirements.txt

2.
git clone https://github.com/THU-MIG/yolov10.git
    - cd yolov10 
    - comment out "onnxruntime-gpu" in the volov10 requirments.txt
    # onnxruntime-gpu
    - pip install -r requirements.txt
    - pip install -e .


4. Execute the main
    - your code should be added to the execute_code()
      - but you can execute the code before
    - as soon as the code is running and watchdog tracking the ./photos dictionary you can move the images (not copy!) L.png, R.png from ./example_photos and both of them get analyzed for the bottles inside them



this is already in the normal requirments.txt
<!-- 
pip install pyqt5==5.15.10 
pip install watchdog==4.0.1
pip install pyfiglet==1.0.2
pip install supervision==0.22.0
pip install qreader==3.14 
-->




