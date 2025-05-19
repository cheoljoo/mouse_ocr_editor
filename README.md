# mouse_ocr_editor
- editor to modify easily when you want to edit your point and rectangle based on ocr results (easyocr)
## getting started
### how to run
- ```uv run main.py```
### uv environment on windows
- [korean reference 출처](https://rudaks.tistory.com/entry/python의-uv-사용법)
- uv install
  - ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```
- ```uv init```
  - Initialize uv environment and create pyproject.toml 
  - Edit description in pyproject.toml
- ```uv run main.py```
  - Add modules after showing module error.
- ```uv add pillow``` for import PIL    (python image library)
- ```uv add easyocr```
- ```uv add opencv-contrib-python``` for import cv2   [reference](https://github.com/astral-sh/uv/issues/11659)
  - If you use opencv-python , opencv-python-headless will be installed because easyocr has dependency with opencv-python-headless. but headless version is not proper for this project.  so you should use opencv-contrib-python.
  - AI suggest the following method.  (but i failed)
      - Remove opencv-python-headless in pyproject.tom
      - ```uv pip compile pyproject.toml -o uv.lock```
      - Edit uv.lock -> remove opencv-python-headless==4.11.0.86
      - ```uv pip sync uv.lock```
  - Alternatives 1: 
    - uv pip install easyocr --no-deps  -> uv pip install opencv-python pillow numpy -> edit pyproject.toml
  - Alternatives 2: if you want to reduce install and loading time when you do not use GPU
    - ```sh
      uv pip install easyocr --no-deps
      uv pip install torch==2.2.2+cpu torchvision==0.17.2+cpu
      uv pip install pillow numpy opencv-python
      uv pip compile pyproject.toml -o uv.lock
      uv pip sync uv.lock
      ```
- ```uv run main.py```
  - ```--restart```  option : You starts if you are 
  - After you can see the picture , you click the points and you draw the rectangle with control key.
    - click point : Point
    - Ctrl + click ~ drag ~ relase : Rectangle
  - easyocr takes a lot of time. so you can skip ocr process when you run second without --restart option.
### how to use the results
- If you need to analyze the points in your png file , first of all you add your png file in root.
- This program will search all png file (*.png) on current path.
- Click_positions.txt is log.
- output directory has the result. input : [ABC].png
  - output/[ABC]/final.png  : this picture includes ocr results and your clicked point and rectangle.
  - output/[ABC]/final.json : this json file includes all info for ocr results and your clicked point and rectangle.
  - output/[ABC]/ocr/[Count].png
    - if [Count] is less than 10000 , it is matched in ocr.
    - if [Count] is more than 10000 , it is picture of your pointed rectangle box.

# info
## should include git for uv
- pyproject.toml 
- uv.lock


