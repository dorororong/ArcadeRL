import tkinter as tk
from PIL import Image, ImageTk
import pyautogui

# 전역 변수
coords = []  # 좌표 저장 리스트

# 클릭 이벤트 처리 함수
def on_click(event):
    global coords
    coords.append((event.x, event.y))  # 클릭한 좌표 저장
    if len(coords) == 2:  # 두 점을 클릭한 경우
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        top, left = min(y1, y2), min(x1, x2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        print(f"'top': {top}, 'left': {left}, 'width': {width}, 'height': {height}")
        root.quit()  # GUI 종료

# 전체 화면 캡처
screenshot = pyautogui.screenshot()
screenshot = screenshot.convert("RGB")  # PIL 이미지로 변환

# Tkinter GUI 설정
root = tk.Tk()
root.title("게임 화면 영역 지정")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

# 캡처된 이미지 표시
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

# 이미지 변환 및 캔버스에 로드
image = ImageTk.PhotoImage(screenshot)
canvas.create_image(0, 0, anchor=tk.NW, image=image)

# 클릭 이벤트 바인딩
canvas.bind("<Button-1>", on_click)

# GUI 실행
root.mainloop()
