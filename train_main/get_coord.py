#!/usr/bin/env python3
import sys
import time
import json
import tkinter as tk
from tkinter import messagebox, simpledialog, Listbox, END
from PIL import Image, ImageTk
import pyautogui

# Globals
current_region = None            # (left, top, width, height)
regions = {}                     # name -> region tuple
overlay = None
canvas = None
start_x_canvas = start_y_canvas = start_x_root = start_y_root = None
rect_id = None
LIVE_INTERVAL_MS = 100           # ms between live frames

def add_new_search_region():
    global overlay, canvas, rect_id, start_x_canvas, start_y_canvas, start_x_root, start_y_root
    start_x_canvas = start_y_canvas = start_x_root = start_y_root = rect_id = None

    overlay = tk.Toplevel(root)
    overlay.attributes('-fullscreen', True)
    overlay.attributes('-alpha',   0.3)
    overlay.attributes('-topmost', True)
    overlay.overrideredirect(True)

    info = tk.Label(
        overlay,
        text="영역을 드래그하여 선택하세요\n(ESC로 취소)",
        bg='white', fg='black',
        font=('Arial', 16)
    )
    info.pack(pady=20)

    canvas = tk.Canvas(overlay, cursor="cross", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    canvas.bind("<ButtonPress-1>", on_region_press)
    canvas.bind("<B1-Motion>",    on_region_drag)
    canvas.bind("<ButtonRelease-1>", on_region_release)
    overlay.bind("<Escape>", cancel_selection)

def on_region_press(event):
    global start_x_canvas, start_y_canvas, start_x_root, start_y_root, rect_id
    start_x_canvas, start_y_canvas = event.x, event.y
    start_x_root, start_y_root = event.x_root, event.y_root
    if rect_id:
        canvas.delete(rect_id)
    rect_id = canvas.create_rectangle(
        start_x_canvas, start_y_canvas,
        start_x_canvas, start_y_canvas,
        outline='red', width=2
    )

def on_region_drag(event):
    global rect_id
    if rect_id:
        canvas.coords(rect_id,
            start_x_canvas, start_y_canvas,
            event.x, event.y
        )

def on_region_release(event):
    global current_region, overlay, rect_id
    end_x_root, end_y_root = event.x_root, event.y_root
    if overlay:
        overlay.destroy()

    if None in (start_x_root, start_y_root):
        return

    left   = min(start_x_root, end_x_root)
    top    = min(start_y_root, end_y_root)
    width  = abs(end_x_root - start_x_root)
    height = abs(end_y_root - start_y_root)

    if width < 5 or height < 5:
        messagebox.showwarning("선택 오류", "너무 작은 영역입니다 (최소 5×5).")
    else:
        current_region = (left, top, width, height)
        status_label.config(
            text=f"선택된 영역 → left:{left}, top:{top}, w:{width}, h:{height}"
        )
    rect_id = None
    update_button_state()

def cancel_selection(event=None):
    global overlay, rect_id
    if overlay:
        overlay.destroy()
    rect_id = None

def save_region():
    global current_region, regions
    if not current_region:
        messagebox.showwarning("저장 오류", "먼저 영역을 선택하세요.")
        return
    name = simpledialog.askstring("영역 저장", "저장할 이름을 입력하세요:")
    if not name:
        return
    regions[name] = current_region
    update_region_listbox()
    status_label.config(text=f"저장됨: {name}")
    update_button_state()

def update_region_listbox():
    listbox_regions.delete(0, END)
    for name in regions:
        listbox_regions.insert(END, name)

def show_live_preview():
    sel = listbox_regions.curselection()
    if not sel:
        messagebox.showwarning("미리보기 오류", "먼저 목록에서 영역을 선택하세요.")
        return
    name = listbox_regions.get(sel[0])
    left, top, w, h = regions[name]

    # get pixel size from entries
    try:
        pw = int(entry_px_width.get())
        ph = int(entry_px_height.get())
        if pw <= 0 or ph <= 0:
            raise ValueError()
    except ValueError:
        messagebox.showerror("입력 오류", "Pixel Width/Height는 양의 정수여야 합니다.")
        return

    win = tk.Toplevel(root)
    win.title(f"Live Preview – {name}")
    img_label = tk.Label(win)
    img_label.pack()
    fps_label = tk.Label(win, text="FPS: 0.0")
    fps_label.pack()

    last_time = time.time()

    def update_frame():
        nonlocal last_time
        now = time.time()
        dt = now - last_time
        fps = 1.0/dt if dt > 0 else 0.0
        last_time = now
        fps_label.config(text=f"FPS: {fps:.1f}")

        img = pyautogui.screenshot(region=(left, top, w, h))
        small = img.resize((pw, ph), Image.BILINEAR)
        disp  = small.resize((w, h), Image.NEAREST)

        photo = ImageTk.PhotoImage(disp)
        img_label.config(image=photo)
        img_label.image = photo

        win.after(LIVE_INTERVAL_MS, update_frame)

    update_frame()

def update_button_state():
    btn_save.config(state="normal" if current_region else "disabled")
    sel = listbox_regions.curselection()
    has_sel = bool(sel)
    btn_show.config(state="normal" if has_sel else "disabled")

# --- Build GUI ---
root = tk.Tk()
root.title("영역 선택기")
root.geometry("420x380")

# Pixel size inputs
px_frame = tk.Frame(root)
px_frame.pack(pady=5)
tk.Label(px_frame, text="Pixel Width:").pack(side="left")
entry_px_width = tk.Entry(px_frame, width=5)
entry_px_width.insert(0, "32")
entry_px_width.pack(side="left", padx=(0,10))
tk.Label(px_frame, text="Pixel Height:").pack(side="left")
entry_px_height = tk.Entry(px_frame, width=5)
entry_px_height.insert(0, "32")
entry_px_height.pack(side="left")

# Control buttons
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

btn_select = tk.Button(
    frame_buttons, text="Select Region",
    command=add_new_search_region, width=12
)
btn_select.pack(side="left", padx=5)

btn_save = tk.Button(
    frame_buttons, text="Save Region",
    command=save_region, width=12, state="disabled"
)
btn_save.pack(side="left", padx=5)

btn_show = tk.Button(
    frame_buttons, text="Show Live",
    command=show_live_preview, width=12, state="disabled"
)
btn_show.pack(side="left", padx=5)

# Status and listbox
status_label = tk.Label(root, text="No region selected", anchor="w")
status_label.pack(fill="x", padx=10)

listbox_regions = Listbox(root, height=5)
listbox_regions.pack(fill="x", padx=10, pady=10)
listbox_regions.bind("<<ListboxSelect>>", lambda e: update_button_state())

root.mainloop()
