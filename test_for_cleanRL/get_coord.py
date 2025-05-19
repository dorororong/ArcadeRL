#!/usr/bin/env python3
import time
import json
import tkinter as tk
from tkinter import messagebox, simpledialog, Listbox, END, filedialog
from PIL import Image, ImageTk
import pyautogui

# ----------- Globals -----------
current_region = None      # (left, top, w, h)
regions = {}               # name -> cfg dict
overlay = None             # used during selection
canvas = None
start_x_canvas = start_y_canvas = None
start_x_root = start_y_root = None
rect_id = None

# ----------- Region Selection -----------
def add_new_search_region():
    global overlay, canvas, rect_id
    global start_x_canvas, start_y_canvas, start_x_root, start_y_root
    start_x_canvas = start_y_canvas = start_x_root = start_y_root = rect_id = None

    overlay = tk.Toplevel(root)
    overlay.attributes('-fullscreen', True)
    overlay.attributes('-alpha', 0.3)
    overlay.attributes('-topmost', True)
    overlay.overrideredirect(True)

    info = tk.Label(
        overlay, text="영역을 드래그하여 선택하세요\n(ESC로 취소)",
        bg='white', fg='black', font=('Arial', 16)
    )
    info.pack(pady=20)

    global canvas
    canvas = tk.Canvas(overlay, cursor="cross", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    canvas.bind("<ButtonPress-1>", on_region_press)
    canvas.bind("<B1-Motion>", on_region_drag)
    canvas.bind("<ButtonRelease-1>", on_region_release)
    overlay.bind("<Escape>", cancel_selection)

def on_region_press(event):
    global start_x_canvas, start_y_canvas, start_x_root, start_y_root, rect_id
    start_x_canvas, start_y_canvas = event.x, event.y
    start_x_root, start_y_root = event.x_root, event.y_root
    if rect_id:
        canvas.delete(rect_id)
    rect_id = canvas.create_rectangle(
        start_x_canvas, start_y_canvas, start_x_canvas, start_y_canvas,
        outline='red', width=2
    )

def on_region_drag(event):
    if rect_id:
        canvas.coords(rect_id, start_x_canvas, start_y_canvas, event.x, event.y)

def on_region_release(event):
    global current_region, overlay, rect_id
    end_x_root, end_y_root = event.x_root, event.y_root
    if overlay:
        overlay.destroy()

    if None in (start_x_root, start_y_root):
        return

    left = min(start_x_root, end_x_root)
    top = min(start_y_root, end_y_root)
    w = abs(end_x_root - start_x_root)
    h = abs(end_y_root - start_y_root)

    if w < 5 or h < 5:
        messagebox.showwarning("선택 오류", "너무 작은 영역입니다 (최소 5×5).")
    else:
        current_region = (left, top, w, h)
        status_label.config(
            text=f"선택된 영역 → left:{left}, top:{top}, w:{w}, h:{h}"
        )
    rect_id = None
    update_button_state()

def cancel_selection(event=None):
    global overlay, rect_id
    if overlay:
        overlay.destroy()
    rect_id = None

# ----------- Region Management -----------
def save_region():
    global current_region
    if not current_region:
        messagebox.showwarning("저장 오류", "먼저 영역을 선택하세요.")
        return
    name = simpledialog.askstring("영역 저장", "저장할 이름을 입력하세요:")
    if not name:
        return
    try:
        pw = int(entry_px_width.get())
        ph = int(entry_px_height.get())
        if pw <= 0 or ph <= 0:
            raise ValueError()
    except ValueError:
        messagebox.showerror("입력 오류", "Pixel Width/Height는 양의 정수여야 합니다.")
        return

    left, top, w, h = current_region
    regions[name] = {
        'left': left, 'top': top,
        'width': w, 'height': h,
        'pixel_width': pw, 'pixel_height': ph
    }
    update_region_listbox()
    status_label.config(text=f"저장됨: {name} ({pw}×{ph})")
    update_button_state()

def load_regions_from_file():
    fname = filedialog.askopenfilename(
        title="JSON 파일 선택",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if not fname:
        return
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'regions' not in data:
            raise KeyError()
    except Exception as e:
        messagebox.showerror("오류", f"파일을 읽을 수 없습니다:\n{e}")
        return

    regions.clear()
    for name, cfg in data['regions'].items():
        if all(k in cfg for k in ("left", "top", "width", "height",
                                  "pixel_width", "pixel_height")):
            regions[name] = cfg
    update_region_listbox()
    status_label.config(text=f"{len(regions)}개 영역 로드됨 – {fname}")
    update_button_state()

def update_region_listbox():
    listbox_regions.delete(0, END)
    for name, cfg in regions.items():
        listbox_regions.insert(END, f"{name} ({cfg['pixel_width']}×{cfg['pixel_height']})")

# ----------- Live Preview -----------
def show_live_preview():
    sel = listbox_regions.curselection()
    if not sel:
        messagebox.showwarning("미리보기 오류", "먼저 목록에서 영역을 선택하세요.")
        return
    key = list(regions.keys())[sel[0]]
    cfg = regions[key]
    left, top, w, h = cfg['left'], cfg['top'], cfg['width'], cfg['height']
    pw, ph = cfg['pixel_width'], cfg['pixel_height']

    # 사용자가 입력한 FPS 가져오기
    try:
        fps = float(entry_fps.get())
        if fps <= 0:
            raise ValueError()
    except ValueError:
        messagebox.showerror("입력 오류", "FPS는 양의 숫자여야 합니다.")
        return
    interval_ms = int(1000 / fps)

    win = tk.Toplevel(root)
    win.title(f"Live Preview – {key} (@{fps:.1f} FPS)")
    img_label = tk.Label(win)
    img_label.pack()
    fps_label = tk.Label(win, text="FPS: 0.0")
    fps_label.pack()

    last_time = time.time()

    def update_frame():
        nonlocal last_time
        now = time.time()
        dt = now - last_time
        actual_fps = 1.0 / dt if dt > 0 else 0.0
        last_time = now
        fps_label.config(text=f"FPS: {actual_fps:.1f}")

        img = pyautogui.screenshot(region=(left, top, w, h))
        small = img.resize((pw, ph), Image.BILINEAR)
        disp = small.resize((w, h), Image.NEAREST)

        photo = ImageTk.PhotoImage(disp)
        img_label.config(image=photo)
        img_label.image = photo

        win.after(interval_ms, update_frame)

    update_frame()

# ----------- Final Save & Exit -----------
def finalize_and_exit():
    if not regions:
        messagebox.showwarning("저장 오류", "저장된 영역이 없습니다.")
        return
    fname = simpledialog.askstring("파일 이름", "저장할 JSON 파일 이름을 입력하세요:")
    if not fname:
        return
    if not fname.lower().endswith('.json'):
        fname += '.json'
    cfg = {'regions': regions}
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    messagebox.showinfo("완료", f"{fname}에 저장되었습니다.")
    root.destroy()

# ----------- Button State -----------
def update_button_state():
    btn_save.config(state="normal" if current_region else "disabled")
    has_sel = bool(listbox_regions.curselection())
    btn_show.config(state="normal" if has_sel else "disabled")
    btn_finalize.config(state="normal" if regions else "disabled")

# ----------- GUI Setup -----------
root = tk.Tk()
root.title("영역 선택기")
root.geometry("520x560")

# Pixel size inputs
px_frame = tk.Frame(root)
px_frame.pack(pady=5)
tk.Label(px_frame, text="Pixel Width:").pack(side="left")
entry_px_width = tk.Entry(px_frame, width=5)
entry_px_width.insert(0, "32")
entry_px_width.pack(side="left", padx=(0, 10))
tk.Label(px_frame, text="Pixel Height:").pack(side="left")
entry_px_height = tk.Entry(px_frame, width=5)
entry_px_height.insert(0, "64")
entry_px_height.pack(side="left", padx=(0, 10))

# FPS input
fps_frame = tk.Frame(root)
fps_frame.pack(pady=5)
tk.Label(fps_frame, text="Preview FPS:").pack(side="left")
entry_fps = tk.Entry(fps_frame, width=5)
entry_fps.insert(0, "4")
entry_fps.pack(side="left")

# Control buttons
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)
btn_select = tk.Button(frame_buttons, text="Select Region",
                       command=add_new_search_region, width=12)
btn_select.pack(side="left", padx=5)
btn_save = tk.Button(frame_buttons, text="Save Region",
                     command=save_region, width=12, state="disabled")
btn_save.pack(side="left", padx=5)
btn_load = tk.Button(frame_buttons, text="Load JSON",
                     command=load_regions_from_file, width=12)
btn_load.pack(side="left", padx=5)

frame_buttons2 = tk.Frame(root)
frame_buttons2.pack(pady=5)
btn_show = tk.Button(frame_buttons2, text="Show Live",
                     command=show_live_preview, width=12, state="disabled")
btn_show.pack(side="left", padx=5)
btn_finalize = tk.Button(frame_buttons2, text="저장 및 종료",
                         command=finalize_and_exit, width=12, state="disabled")
btn_finalize.pack(side="left", padx=5)

# Status and listbox
status_label = tk.Label(root, text="No region selected", anchor="w")
status_label.pack(fill="x", padx=10)
listbox_regions = Listbox(root, height=10)
listbox_regions.pack(fill="x", padx=10, pady=10)
listbox_regions.bind("<<ListboxSelect>>", lambda e: update_button_state())

root.mainloop()
