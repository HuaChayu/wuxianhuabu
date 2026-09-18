#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 N 张竖版角色立绘拼成一张横版 864x480 的合影（供 H3 FL2VA 首/尾帧使用）。

- 每格固定宽 288 / 高 480（864/3），人物等比缩放到高 480 后居中放进白底格。
- B 图在 A 图基础上中心放大 1.08 再裁回 864x480，制造轻微推近，避免首尾同图导致完全静止。
"""
import os
from PIL import Image

OUT = "/Users/huachayui/Desktop/无限画布/model/minimax h3/assets"
W, H = 864, 480

ROLES = [
    ("2.png", "少女"),   # 薄荷绿新中式短裙
    ("3.png", "狐耳男"),  # 金发狐耳
    ("4.png", "银发男"),  # 银冠黑袍
]
SRC = "/Users/huachayui/Downloads/角色"

os.makedirs(OUT, exist_ok=True)
slot_w = W // len(ROLES)
canvas = Image.new("RGB", (W, H), (255, 255, 255))

for i, (fn, name) in enumerate(ROLES):
    im = Image.open(os.path.join(SRC, fn)).convert("RGB")
    scale = H / im.height
    nw, nh = max(1, round(im.width * scale)), H
    im = im.resize((nw, nh), Image.LANCZOS)
    if nw > slot_w:
        left = (nw - slot_w) // 2
        im = im.crop((left, 0, left + slot_w, H))
    cell = Image.new("RGB", (slot_w, H), (255, 255, 255))
    cell.paste(im, ((slot_w - im.width) // 2, 0))
    canvas.paste(cell, (i * slot_w, 0))
    print(f"slot{i}: {name} {fn} -> {im.size}")

a_path = os.path.join(OUT, "three_shot_a.png")
canvas.save(a_path)
print("A:", a_path, canvas.size)

# B：中心放大 1.08 后裁回原尺寸（轻微推近，避免首尾完全同图）
z = 1.08
big = canvas.resize((round(W * z), round(H * z)), Image.LANCZOS)
left, top = (big.width - W) // 2, (big.height - H) // 2
canvas_b = big.crop((left, top, left + W, top + H))
b_path = os.path.join(OUT, "three_shot_b.png")
canvas_b.save(b_path)
print("B:", b_path, canvas_b.size)
