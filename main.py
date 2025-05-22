from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
import os
import time
import math
import json
import copy
import argparse
import shutil
from pprint import pprint , pformat
import threading

if os.name != 'nt':
    print('os is not windows')
    print(os.name)
    quit(4)

# 현재 이 코드는 vscode에서 수행하는 것이다. 
# sys.prefix  으로 환경 위치 확인
# vscode F5 수행으로 인하여 실행 위치가 git의 root가 된다.  os.getcwd() 으로 확인
# TERMINAL로 들어간 위치가 pwd가 되는 것이다.

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description= 'ocr & point editor',
)
parser.add_argument( '--restart', default=False , action="store_true" , help="restart after deleting output directory")

args = parser.parse_args()



# 저장할 파일
output_file = "click_positions.txt"
start_time = time.time()
def log_elapsed_time(argMsg=''):
    elapsed = round(time.time() - start_time, 2)
    if argMsg.strip():
        message = f"⏱ Elapsed time: {elapsed} seconds: msg : {argMsg}"
    else:
        message = f"⏱ Elapsed time: {elapsed} seconds"
    print(message)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# 로딩 플래그
loading = True
# 점 찍는 함수
def show_loading_dots():
    while loading:
        print(".", end="", flush=True)
        time.sleep(1)
# 스레드 시작
loading_thread = threading.Thread(target=show_loading_dots)

print("Python Executable:", sys.executable)
print("sys.prefix:", sys.prefix)
print('pwd', os.getcwd())
log_elapsed_time()
import cv2
print('cv2',cv2.__version__)

def find_nearest_ocr_text(point, ocr_results,rec_results):
    """주어진 point와 가장 가까운 OCR 텍스트 반환"""
    px, py = point
    nearest_text = None
    min_distance = float("inf")
    ocrIndex = None
    for idx,(bbox, text, confidence) in enumerate(ocr_results):
        # 중심점 계산
        (tl, tr, br, bl) = bbox
        cx = int((tl[0] + br[0]) / 2)
        cy = int((tl[1] + br[1]) / 2)
        dist = math.hypot(px - cx, py - cy)

        if dist < min_distance:
            min_distance = dist
            nearest_text = text
            ocrIndex = idx

    min_distance = float("inf")
    recIndex = None
    for idx,v in enumerate(rec_results):
        if v['type'] != 'rectangle':
            continue
        # 중심점 계산
        cx = v['center'][0]
        cy = v['center'][1]
        dist = math.hypot(px - cx, py - cy)

        if dist < min_distance:
            min_distance = dist
            recIndex = idx

    return ocrIndex ,recIndex


def check_opencv_gui_support():
    try:
        # 임시 창 열기 시도
        cv2.namedWindow("test_check_window")
        cv2.destroyWindow("test_check_window")
    except cv2.error as e:
        if "The function is not implemented" in str(e):
            print("\nerror: ❌ OpenCV GUI is not available")
            print("➡ 'opencv-python-headless' 버전이 설치되어 있을 가능성이 큽니다.")
            print("💡 solution:")
            print("   1. 'headless' 버전 제거: pip3 uninstall opencv-python-headless")
            print("   2. GUI 지원 버전 설치: pip3 uninstall openvc-python ->  pip3 install opencv-python\n")
            sys.exit(1)
        else:
            raise e  # 다른 에러는 그대로 올림

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy(i) for i in obj]  # JSON은 튜플도 리스트로
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def mouse_callback(event, x, y, flags, param):   # 마우스 이벤트 콜백 함수
    global drawing, start_point, end_point, img

    if event == cv2.EVENT_RBUTTONDOWN and not (flags & cv2.EVENT_FLAG_CTRLKEY):
        print(f"Clicked at: R({x}, {y})")
        

        # 점 그리기 (빨간 원)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # 텍스트 표시
        cv2.putText(img, f"R({x},{y})", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 위치 저장
        your_clicks.append( {'type':'rightPoint','location':[x,y],'center':[x,y]}  )

        # 파일에 저장 (append 모드)
        with open(output_file, "a") as f:
            f.write(f"R{x},{y}\n")
        return

    if event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_CTRLKEY):
        print(f"Clicked at: L({x}, {y})")
        

        # 점 그리기 (빨간 원)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # 텍스트 표시
        cv2.putText(img, f"L({x},{y})", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 위치 저장
        your_clicks.append( {'type':'leftPoint','location':[x,y],'center':[x,y]}  )

        # 파일에 저장 (append 모드)
        with open(output_file, "a") as f:
            f.write(f"L{x},{y}\n")
        return

    # Ctrl + 마우스 왼쪽 버튼 눌렀을 때
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)
        temp_img = img.copy()
        cv2.rectangle(temp_img, start_point, end_point, (255, 0, 0), 2)
        cv2.imshow("Image", temp_img)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end_point = (x, y)

        # 사각형 그리기
        cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)

        # 텍스트 표시
        text_pos = (start_point[0] + 10, start_point[1] + 10)
        cv2.putText(img, f"{start_point}->{end_point}", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 중심점 기준으로 OCR 텍스트 검색
        center_x = int((start_point[0] + end_point[0]) / 2)
        center_y = int((start_point[1] + end_point[1]) / 2)


        your_clicks.append( {'type':'rectangle','location':[start_point,end_point],'center':[center_x,center_y]}  )

        # 위치 저장
        with open(output_file, "a") as f:
            f.write(f"{start_point[0]},{start_point[1]},{end_point[0]},{end_point[1]}\n")


# 실행 초기에 호출
log_elapsed_time('opencw_python version check starts')
check_opencv_gui_support()
log_elapsed_time('opencw_python version check ends')

log_elapsed_time('hangul font load starts')
# ✅ 한글 지원 폰트 경로 (윈도우 기준 예: 맑은 고딕)
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 20)
log_elapsed_time('hangul font load ends')

files = os.listdir('.')
ansFiles = []
for file in files:
    if file.endswith('.png'):
        ansFiles.append(file)
print('fileList:',ansFiles)

outputdir = 'output'
if args.restart:
    shutil.rmtree(outputdir, ignore_errors=True)
shutil.rmtree(output_file, ignore_errors=True)

your_code = '''
import pyautogui
import cv2


'''
your_md = ''
states = ['INIT']

for img_path in ansFiles:
    your_clicks = []
    ocr_crop_dir = outputdir + '/' + img_path.replace('.png','')
    pngdir = ocr_crop_dir + '/ocr'
    os.makedirs(pngdir, exist_ok=True)

    funcDef = ''.join([char for char in img_path.replace('.png','') if char.isalnum()])
    if not funcDef.upper() in states:
        states.append(funcDef.upper())

    # 이미지 로드
    originImg = cv2.imread(img_path)
    if originImg is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")
    log_elapsed_time('image load : {i}'.format(i=img_path))

    img = originImg.copy()
    ocrMaxCount = 10000
    if not os.path.exists(os.path.join(ocr_crop_dir, "intermediateOcr.png")):
        if loading == True:
            log_elapsed_time('easyocr load starts')
            loading_thread.start()
            print('loading : easyocr module')
            import easyocr
            loading = False
            loading_thread.join()
            print('loading done')
            log_elapsed_time('easyocr load ends')

        # easyocr
        reader = easyocr.Reader(['en', 'ko'], gpu=False)
        ocr_results = reader.readtext(img)
        # print('ocr_results',ocr_results)
        log_elapsed_time('get ocr results')

        for idx, (bbox, text, confidence) in enumerate(ocr_results):
            # 좌표 정리
            (tl, tr, br, bl) = bbox
            x_coords = [int(pt[0]) for pt in [tl, tr, br, bl]]
            y_coords = [int(pt[1]) for pt in [tl, tr, br, bl]]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # ROI 잘라내기 (원본 이미지에서)
            roi = img[y_min:y_max, x_min:x_max]

            # 저장
            output_path = os.path.join(pngdir, f"{idx}.png")
            cv2.imwrite(output_path, roi)
        log_elapsed_time(f'make png file each ocr detection in {pngdir}')

        # cv2 -> PIL로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # 보라색 RGB = (255, 0, 255)
        for (bbox, text, confidence) in ocr_results:
            (tl, tr, br, bl) = bbox
            tl, br = tuple(map(int, tl)), tuple(map(int, br))
            draw.rectangle([tl, br], outline=(255, 0, 255), width=2)
            draw.text((tl[0], tl[1] - 20), text, fill=(255, 0, 255), font=font)
        # PIL -> OpenCV로 다시 변환
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # 저장
        cv2.imwrite(os.path.join(ocr_crop_dir, "intermediateOcr.png"), img)
        log_elapsed_time('create file:'+os.path.join(ocr_crop_dir, "intermediateOcr.png"))
    else:
        img = cv2.imread(os.path.join(ocr_crop_dir, "intermediateOcr.png"))
        ocr_results = []
        jsons = json.dumps(convert_numpy({'ocr_results':ocr_results,'your_click':your_clicks}),indent=4)
        if os.path.exists(os.path.join(ocr_crop_dir, "final.json")):
            with open(os.path.join(ocr_crop_dir,"final.json"),'r',encoding="utf-8") as infile:
                print('read :', os.path.join(ocr_crop_dir,"final.json"))
                jsons = json.load(infile)
                ocr_results = jsons['ocr_results']
                your_clicks = jsons['your_click']

    # 좌표 저장용
    clone = img.copy()
    # 사각형 그리기 상태 변수
    drawing = False
    start_point = (-1, -1)
    end_point = (-1, -1)

    log_elapsed_time('opencv windows setting starts')
    # OpenCV 윈도우 설정
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Image", 0, 0)
    cv2.setMouseCallback("Image", mouse_callback)
    log_elapsed_time('opencv windows setting ends')

    # 메인 루프
    while True:
        if not drawing:
            # 이미지 크기 가져오기
            height, width = img.shape[:2]
            cv2.resizeWindow("Image", width, height)
            if your_clicks:
                for your_click in your_clicks:
                    if your_click['type'] == 'leftPoint' or your_click['type'] == 'rightPoint':
                        x,y = your_click['location']
                        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(img, f"L({x},{y})", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    elif your_click['type'] == 'rectangle':
                        start_point = your_click['location'][0]
                        end_point = your_click['location'][1]
                        cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
                        text_pos = (start_point[0] + 10, start_point[1] + 10)
                        cv2.putText(img, f"{start_point}->{end_point}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(img, f"L({x},{y})", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Image", img)
                        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            img = clone.copy()
            cv2.imshow("Image", img)
            your_clicks = []
        elif key == 27:
            finalpng = os.path.join(ocr_crop_dir,"final.png")
            cv2.imwrite(os.path.join(ocr_crop_dir,"final.png"), img)  # 최종 이미지 저장
            log_elapsed_time(f"최종 이미지가 {finalpng}로 저장되었습니다.")
            copied_your_clicks = copy.deepcopy(your_clicks)
            count = ocrMaxCount + 100
            for v in your_clicks:
                ocridx,youridx = find_nearest_ocr_text(v['center'], ocr_results,your_clicks)
                if ocridx != None:
                    (bbox, text, confidence) = ocr_results[ocridx]
                    print("📌 your click과 가장 가까운 OCR 텍스트:", v , '~~', ocridx, bbox , text)
                    v['nearest_ocr'] = ocr_results[ocridx]
                if youridx != None:
                    yourClickDict = copied_your_clicks[youridx]
                    print("📌 your click과 가장 가까운 your drag box index:", v , '~~' , youridx , yourClickDict, yourClickDict['location'])
                    v['nearest_rec'] = yourClickDict
                if v['type'] == 'rectangle':
                    x_coords = [int(pt[0]) for pt in v['location']]
                    y_coords = [int(pt[1]) for pt in v['location']]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    print('v',v['location'],x_min, x_max,y_min, y_max)

                    # ROI 잘라내기 (원본 이미지에서)
                    if y_min == y_max:
                        y_max = y_min + 1
                    if x_min == x_max:
                        x_max = x_min + 1
                    roi = originImg[y_min:y_max, x_min:x_max]

                    # 저장
                    output_path = os.path.join(pngdir, f"{count}.png")
                    cv2.imwrite(output_path, roi)
                    v['image'] = output_path
                    count += 1
            break

    cv2.destroyAllWindows()
    log_elapsed_time('opencv windows destroyed')

    # 문자열로 변환
    data = {'ocr_results':ocr_results,'your_click':your_clicks}
    pretty_string = pformat(data)
    # 파일에 저장
    with open(os.path.join(ocr_crop_dir,"final2.json"), "w", encoding="utf-8") as f:
        print('write:',os.path.join(ocr_crop_dir,"final2.json"))
        f.write(pretty_string)

    # pprint(ocr_results)
    with open(os.path.join(ocr_crop_dir,"ocr_results.json"),'w',encoding="utf-8") as outfile:
        print('write :', os.path.join(ocr_crop_dir,"ocr_results.json"))
        jsons = json.dumps(convert_numpy(ocr_results),indent=4)
        outfile.write(jsons)

    with open(os.path.join(ocr_crop_dir,"final.json"),'w',encoding="utf-8") as outfile:
        print('write :', os.path.join(ocr_crop_dir,"final.json"))
        jsons = json.dumps(convert_numpy({'ocr_results':ocr_results,'your_click':your_clicks}),indent=4)
        # jsons = json.dumps({'ocr_results':ocr_results,'your_click':your_clicks},indent=4)
        outfile.write(jsons)
    your_code += '''def f{img}Png():\n'''.format(img=funcDef)
    spaces = 4
    recCount = 1
    local_check_md = ''
    local_point_md = ''
    code_flag = False
    for v in your_clicks:
        if v['type'] == 'rectangle':
            your_code += ' '*spaces +   '''center{c},left{c} = findImageMoveToCenter('{image}')\n'''.format(image=v['image'],c=recCount)
            your_code += ' '*spaces +   '''if center{c}:\n'''
            your_code += ' '*spaces +   '''    pass\n'''
            your_code += '\n'
            local_check_md += '- check rectangle whether it exists or not when we enter in {state} STATE.\n'.format(state=funcDef.upper())
            local_check_md += '    - position:{position} ![]({img})\n'.format(position=v['location'],img=v['image'])
            recCount += 1
            code_flag = True
        elif v['type'] == 'leftPoint':
            your_code += ' '*spaces +   '''pyautogui.moveTo({position})\n'''.format(position=v['location'])
            your_code += ' '*spaces +   '''pyautogui.click()\n'''
            your_code += '\n'
            local_point_md += '- click this position: {position} , STATE {state}\n'.format(state=funcDef.upper(),position=v['location'])
            code_flag = True
        elif v['type'] == 'rightPoint':
            your_code += ' '*spaces +   '''pyautogui.moveTo({position})\n'''.format(position=v['location'])
            your_code += ' '*spaces +   '''pyautogui.click()\n'''
            your_code += ' '*spaces +   '''pyautogui.write('{text}'.format(text='your_text'), interval=0.15)   # type with interval 0.15 sec. you need return key when you want to type return\n'''
            local_point_md += '- click this position: {position} , STATE {state} and write the TEXT\n'.format(state=funcDef.upper(),position=v['location'])
            your_code += '\n'
            code_flag = True
    if code_flag == False:
        your_code += ' '*spaces +   '''pass\n'''
    your_code += '\n'
    if local_check_md or local_point_md:
        your_md += '# STATE : {state}\n'.format(state=funcDef.upper())
        your_md += '- python function name : {fn}\n'.format(fn=funcDef+'Png')
        your_md += '- image path : {ip}\n'.format(ip=img_path)
        your_md += '    - ![]({ip})\n'.format(ip=img_path)
        if local_check_md:
            your_md += local_check_md
        if local_point_md:
            your_md += local_point_md
        your_md += '- final image path : {ip}\n'.format(ip=os.path.join(ocr_crop_dir,"final.png"))
        your_md += '    - ![]({ip})\n'.format(ip=os.path.join(ocr_crop_dir,"final.png"))

your_code += '''STATES = [\n'''
for i,state in enumerate(states):
    if i == 0:
        your_code += ' '*spaces + """'{state}'\n""".format(state=state)
    else:
        your_code += ' '*spaces + """, '{state}'\n""".format(state=state)
your_code += ''']\n'''

with open(os.path.join(".","your.py"),'w',encoding="utf-8") as outfile:
    print('write :', os.path.join(".","your.py"))
    outfile.write(your_code)
with open(os.path.join(".","your.md"),'w',encoding="utf-8") as outfile:
    print('write :', os.path.join(".","your.md"))
    outfile.write(your_md)




# todo
# 위의 ocr로 text를 뽑은 내용들에 대해서 png image를 만들어 달라. (Done)
# 여러개의 file에 대해서 연속으로 작업하게 해 달라. (Done)
# ocr_results에 rectangle로 내가 정의한 내용까지 추가해 달라. (Done)
# rectangle로 지정한 것에 대해서 초기화면에서 해당 부분의 image를 추출하고 png directory에 번호를 추가해서 넣어주고 해당 정보를 your_clicks 정보에 추가 한다. (Done)
# uv를 이용할 것
