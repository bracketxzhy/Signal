from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


SCU_GREEN = (8, 79, 52)
SCU_LIGHT = (223, 233, 227)
SCU_DOT = (35, 137, 79)
SCU_TEXT = (41, 63, 70)
GOLD = (239, 180, 62)
TEAL = (28, 174, 170)
BLUE = (62, 137, 210)
RED = (220, 75, 70)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F12 = font(22)
F14 = font(26)
F16 = font(30)
F18 = font(34)
F22 = font(42, True)


def canvas(w: int = 1400, h: int = 900, bg=(255, 255, 255)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)


def trim_whitespace(img: Image.Image, pad: int = 20) -> Image.Image:
    rgb = img.convert("RGB")
    pix = rgb.load()
    w, h = rgb.size
    min_x, min_y, max_x, max_y = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            r, g, b = pix[x, y]
            if min(r, g, b) < 246:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < 0:
        return img
    return img.crop((
        max(0, min_x - pad),
        max(0, min_y - pad),
        min(w, max_x + pad + 1),
        min(h, max_y + pad + 1),
    ))


def save(img: Image.Image, name: str, pad: int = 20) -> None:
    trim_whitespace(img, pad=pad).save(OUT / name, optimize=True)


def arrow(draw: ImageDraw.ImageDraw, start, end, fill=SCU_GREEN, width=5) -> None:
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    a = math.atan2(y2 - y1, x2 - x1)
    s = 18
    pts = [
        (x2, y2),
        (x2 - s * math.cos(a - 0.45), y2 - s * math.sin(a - 0.45)),
        (x2 - s * math.cos(a + 0.45), y2 - s * math.sin(a + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def plot_axes(draw, box, xlabel="", ylabel="", color=(180, 190, 190)) -> None:
    x0, y0, x1, y1 = box
    draw.line([(x0, y1), (x1, y1)], fill=color, width=2)
    draw.line([(x0, y0), (x0, y1)], fill=color, width=2)
    if xlabel:
        draw.text((x1 - 30, y1 + 8), xlabel, fill=SCU_TEXT, font=F12)
    if ylabel:
        draw.text((x0 - 25, y0 - 5), ylabel, fill=SCU_TEXT, font=F12)


def map_xy(box, x, y, xr, yr):
    x0, y0, x1, y1 = box
    u = x0 + (x - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
    v = y1 - (y - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)
    return u, v


def polyline(draw, box, pts, xr, yr, fill=SCU_GREEN, width=4):
    mapped = [map_xy(box, x, y, xr, yr) for x, y in pts]
    draw.line(mapped, fill=fill, width=width, joint="curve")


def square_wave_approximation() -> None:
    img, draw = canvas(1500, 760)
    boxes = [(90, 45, 1410, 225), (90, 290, 1410, 470), (90, 535, 1410, 715)]
    terms = [1, 5, 15]
    xr = (-math.pi, math.pi)
    for box, n_terms in zip(boxes, terms):
        plot_axes(draw, box, "x", "f(x)")
        square = []
        approx = []
        for i in range(900):
            x = xr[0] + i / 899 * (xr[1] - xr[0])
            square.append((x, 1 if x >= 0 else -1))
            y = 0.0
            for k in range(n_terms):
                n = 2 * k + 1
                y += math.sin(n * x) / n
            approx.append((x, 4 / math.pi * y))
        polyline(draw, box, square, xr, (-1.7, 1.7), fill=(170, 170, 170), width=4)
        polyline(draw, box, approx, xr, (-1.7, 1.7), fill=SCU_DOT, width=5)
        draw.text((box[0] + 15, box[1] + 12), f"N = {n_terms} odd term{'s' if n_terms > 1 else ''}", fill=SCU_GREEN, font=F18)
    save(img, "fourier_square_wave_approx.png")


def polar_dumbbell() -> None:
    img, draw = canvas(1000, 760)
    cx, cy, scale = 500, 380, 255
    for r in [0.25, 0.5, 0.75, 1.0]:
        draw.ellipse((cx - scale * r, cy - scale * r, cx + scale * r, cy + scale * r), outline=(220, 228, 224), width=2)
    pts = []
    for i in range(721):
        th = 2 * math.pi * i / 720
        r = abs(math.cos(th)) * (0.78 + 0.14 * math.cos(4 * th))
        pts.append((cx + scale * r * math.cos(th), cy - scale * r * math.sin(th)))
    draw.polygon(pts, fill=(202, 235, 221), outline=SCU_GREEN)
    draw.line(pts + [pts[0]], fill=SCU_GREEN, width=6, joint="curve")
    arrow(draw, (cx - scale - 55, cy), (cx + scale + 65, cy), fill=(95, 116, 116), width=4)
    arrow(draw, (cx, cy + scale + 55), (cx, cy - scale - 65), fill=(95, 116, 116), width=4)
    draw.text((cx + scale + 75, cy - 16), "x", fill=SCU_TEXT, font=F14)
    draw.text((cx + 12, cy - scale - 92), "y", fill=SCU_TEXT, font=F14)
    th = math.radians(37)
    r = abs(math.cos(th)) * (0.78 + 0.14 * math.cos(4 * th))
    arrow(draw, (cx, cy), (cx + scale * r * math.cos(th), cy - scale * r * math.sin(th)), fill=RED, width=4)
    draw.arc((cx - 75, cy - 75, cx + 75, cy + 75), 323, 360, fill=BLUE, width=5)
    draw.text((cx + 155, cy - 150), "r(theta)", fill=RED, font=F18)
    draw.text((cx + 85, cy - 45), "theta", fill=BLUE, font=F16)
    save(img, "polar_dumbbell_r_costheta.png")


def rotate(p, yaw=-0.75, pitch=0.6):
    x, y, z = p
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, z = cy * x + sy * z, -sy * x + cy * z
    cp, sp = math.cos(pitch), math.sin(pitch)
    y, z = cp * y - sp * z, sp * y + cp * z
    return x, y, z


def project(p, center=(700, 460), scale=135, yaw=-0.72):
    x, y, z = p
    ca, sa = math.cos(yaw), math.sin(yaw)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return center[0] + scale * xr, center[1] + scale * (0.48 * yr - z), yr + 0.2 * z


def color_mix(v, lo=(24, 128, 170), mid=(48, 185, 120), hi=(230, 70, 45)):
    v = max(0, min(1, v))
    if v < 0.5:
        t = v * 2
        return tuple(int(lo[i] * (1 - t) + mid[i] * t) for i in range(3))
    t = (v - 0.5) * 2
    return tuple(int(mid[i] * (1 - t) + hi[i] * t) for i in range(3))


def draw_surface(draw, func, box=None, center=(700, 480), scale=120, nx=55, ny=55, wire=True, zrange=None):
    polys = []
    xs = [-3 + 6 * i / (nx - 1) for i in range(nx)]
    ys = [-3 + 6 * j / (ny - 1) for j in range(ny)]
    vals = [[func(x, y) for y in ys] for x in xs]
    if zrange is None:
        flat = [v for row in vals for v in row]
        zrange = (min(flat), max(flat))
    def pr(x, y, z):
        u, v, d = project((x, y, z), center=center, scale=scale)
        return (u, v), d
    for i in range(nx - 1):
        for j in range(ny - 1):
            corners = [(xs[i], ys[j], vals[i][j]), (xs[i + 1], ys[j], vals[i + 1][j]),
                       (xs[i + 1], ys[j + 1], vals[i + 1][j + 1]), (xs[i], ys[j + 1], vals[i][j + 1])]
            pts, depth = [], 0
            for c in corners:
                p, d = pr(*c)
                pts.append(p)
                depth += d
            zavg = sum(c[2] for c in corners) / 4
            t = (zavg - zrange[0]) / max(1e-6, zrange[1] - zrange[0])
            polys.append((depth / 4, pts, color_mix(t)))
    for _, pts, col in sorted(polys, key=lambda x: x[0]):
        draw.polygon(pts, fill=col)
        if wire:
            draw.line(pts + [pts[0]], fill=(70, 100, 80), width=1)


def draw_3d_axes(draw, center=(700, 480), scale=120, x0=-3.25, y0=-3.25, z0=-1.35, length=6.6, height=3.15) -> None:
    def p(x, y, z):
        u, v, _ = project((x, y, z), center=center, scale=scale)
        return u, v

    origin = p(x0, y0, z0)
    x_end = p(x0 + length, y0, z0)
    y_end = p(x0, y0 + length, z0)
    z_end = p(x0, y0, z0 + height)
    arrow(draw, origin, x_end, fill=(45, 58, 62), width=4)
    arrow(draw, origin, y_end, fill=(45, 58, 62), width=4)
    arrow(draw, origin, z_end, fill=(45, 58, 62), width=4)
    draw.text((x_end[0] + 6, x_end[1] - 6), "x", fill=SCU_TEXT, font=F14)
    draw.text((y_end[0] + 6, y_end[1] - 6), "y", fill=SCU_TEXT, font=F14)
    draw.text((z_end[0] + 6, z_end[1] - 8), "z", fill=SCU_TEXT, font=F14)


def two_variable_surface() -> None:
    img, draw = canvas(980, 760)
    def f(x, y):
        return (
            1.4 * math.exp(-((x - 1.2) ** 2 + (y - 0.8) ** 2) / 0.7)
            + 0.8 * math.exp(-((x + 0.6) ** 2 + (y + 1.0) ** 2) / 0.45)
            - 1.2 * math.exp(-((x + 1.6) ** 2 + (y - 1.1) ** 2) / 0.55)
            + 0.2 * math.sin(2 * x + y)
        )
    draw_surface(draw, f, center=(485, 405), scale=92, nx=62, ny=62, zrange=(-1.4, 1.7))
    draw_3d_axes(draw, center=(485, 405), scale=92)
    save(img, "two_variable_surface.png")


def two_variable_basis_functions() -> None:
    modes = [
        (1, 1, "two_variable_basis_n1.png"),
        (1, 2, "two_variable_basis_n2.png"),
        (1, 3, "two_variable_basis_n3.png"),
        (1, 4, "two_variable_basis_n4.png"),
    ]
    for m, n, name in modes:
        img, draw = canvas(760, 560)

        def basis(x, y, mx=m, ny=n):
            return 0.9 * math.cos(mx * x) * math.cos(ny * y)

        draw_surface(
            draw,
            basis,
            center=(380, 305),
            scale=58,
            nx=54,
            ny=54,
            zrange=(-1.0, 1.0),
        )
        draw_3d_axes(
            draw,
            center=(380, 305),
            scale=58,
            x0=-3.2,
            y0=-3.2,
            z0=-1.05,
            length=6.5,
            height=2.45,
        )
        save(img, name, pad=8)


def two_variable_fit_grid() -> None:
    img, draw = canvas(1200, 820)
    def target(x, y):
        return (
            1.4 * math.exp(-((x - 1.2) ** 2 + (y - 0.8) ** 2) / 0.7)
            + 0.8 * math.exp(-((x + 0.6) ** 2 + (y + 1.0) ** 2) / 0.45)
            - 1.2 * math.exp(-((x + 1.6) ** 2 + (y - 1.1) ** 2) / 0.55)
            + 0.2 * math.sin(2 * x + y)
        )
    def approx(x, y, n):
        val = 0.0
        if n >= 1:
            val += 0.65 * math.exp(-((x - 1.2) ** 2 + (y - 0.8) ** 2) / 1.35)
        if n >= 2:
            val += 0.65 * math.exp(-((x + 0.6) ** 2 + (y + 1.0) ** 2) / 0.95)
        if n >= 4:
            val -= 1.0 * math.exp(-((x + 1.6) ** 2 + (y - 1.1) ** 2) / 0.85)
        if n >= 8:
            val = target(x, y)
        return val
    labels = [("N = 1", 1), ("N = 2", 2), ("N = 4", 4), ("N = 8", 8)]
    centers = [(310, 220), (890, 220), (310, 610), (890, 610)]
    for (label, n), center in zip(labels, centers):
        draw_surface(draw, lambda x, y, k=n: approx(x, y, k), center=center, scale=42, nx=36, ny=36, zrange=(-1.4, 1.7))
        draw_3d_axes(draw, center=center, scale=42)
        draw.text((center[0] - 265, center[1] - 175), label, fill=SCU_TEXT, font=F14)
    save(img, "two_variable_fit_grid.png")


def spherical_coordinates() -> None:
    img, draw = canvas(980, 760)
    cx, cy = 500, 385
    radius = 235
    arrow(draw, (cx - 340, cy + 145), (cx + 375, cy - 155), fill=(70, 82, 86), width=4)
    arrow(draw, (cx - 310, cy - 135), (cx + 350, cy + 165), fill=(70, 82, 86), width=4)
    arrow(draw, (cx, cy + 315), (cx, cy - 325), fill=(70, 82, 86), width=4)
    draw.text((cx + 385, cy - 176), "x", fill=SCU_TEXT, font=F14)
    draw.text((cx + 360, cy + 172), "y", fill=SCU_TEXT, font=F14)
    draw.text((cx + 12, cy - 350), "z", fill=SCU_TEXT, font=F14)
    for r in range(radius, 0, -1):
        shade = int(232 - 80 * (r / radius))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(shade, shade + 5, shade + 8))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=SCU_GREEN, width=4)
    draw.ellipse((cx - radius, cy - 52, cx + radius, cy + 52), outline=(120, 160, 155), width=3)
    th, ph = math.radians(48), math.radians(32)
    x = radius * math.sin(th) * math.cos(ph)
    y = -radius * math.cos(th)
    px, py = cx + x, cy + y
    arrow(draw, (cx, cy), (px, py), fill=RED, width=5)
    draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=RED)
    draw.text((px + 18, py - 18), "point", fill=RED, font=F14)
    draw.text(((cx + px) / 2 + 15, (cy + py) / 2 - 35), "r", fill=RED, font=F18)
    draw.arc((cx - 120, cy - 120, cx + 120, cy + 120), 270, 318, fill=BLUE, width=5)
    draw.text((cx + 70, cy - 125), "theta", fill=BLUE, font=F18)
    draw.arc((cx - 160, cy - 42, cx + 160, cy + 42), 178, 212, fill=GOLD, width=5)
    draw.text((cx - 190, cy + 35), "phi", fill=GOLD, font=F18)
    save(img, "spherical_coordinates_theta_phi_r.png")


def lobe_surface(l: int, m: int, positive=(93, 174, 190), negative=(245, 190, 78), size=(240, 210)) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    polys = []
    nt, np = 26, 42
    for i in range(nt - 1):
        t0 = math.pi * i / (nt - 1)
        t1 = math.pi * (i + 1) / (nt - 1)
        for j in range(np - 1):
            p0 = 2 * math.pi * j / (np - 1)
            p1 = 2 * math.pi * (j + 1) / (np - 1)
            verts = []
            signs = []
            depth = 0
            for th, ph in [(t0, p0), (t1, p0), (t1, p1), (t0, p1)]:
                val = math.cos(l * th) * math.cos(m * ph) if m else math.cos(l * th)
                r = 0.55 + 0.38 * abs(val)
                x = r * math.sin(th) * math.cos(ph)
                y = r * math.sin(th) * math.sin(ph)
                z = r * math.cos(th)
                u, v, d = project((x, y, z), center=(w / 2, h / 2), scale=75)
                verts.append((u, v))
                signs.append(val)
                depth += d
            base = positive if sum(signs) >= 0 else negative
            shade = 0.78 + 0.22 * max(-1, min(1, depth / 4))
            col = tuple(int(c * shade) for c in base) + (255,)
            polys.append((depth / 4, verts, col))
    for _, pts, col in sorted(polys, key=lambda x: x[0]):
        draw.polygon(pts, fill=col)
    return img.filter(ImageFilter.SMOOTH)


def sh_basis_grid() -> None:
    img, draw = canvas(1120, 780)
    rows = [
        ("L = 0", [("m = 0", (0, 0))]),
        ("L = 1", [("m = -1", (1, 1)), ("m = 0", (1, 0)), ("m = 1", (1, 1))]),
        ("L = 2", [("m = -2", (2, 2)), ("m = -1", (2, 1)), ("m = 0", (2, 0)), ("m = 1", (2, 1)), ("m = 2", (2, 2))]),
    ]
    for row_idx, (row_label, items) in enumerate(rows):
        y = 20 + row_idx * 245
        draw.text((20, y + 78), row_label, fill=SCU_TEXT, font=F14)
        total_w = len(items) * 190
        start_x = 120 + (900 - total_w) / 2
        for col_idx, (lab, (l, m)) in enumerate(items):
            x = int(start_x + col_idx * 190)
            patch = lobe_surface(l, m, size=(185, 160))
            img.paste(patch, (x, y), patch)
            draw.text((x + 48, y + 158), lab, fill=SCU_TEXT, font=F12)
    save(img, "sh_basis_grid.png")


def sh_coefficient_projection() -> None:
    img, draw = canvas(1100, 430)
    positions = [(90, 55), (340, 55), (590, 55), (840, 55)]
    labels = ["signal", "0.72 Y0", "-0.35 Y1", "0.18 Y2"]
    params = [(3, 2), (0, 0), (1, 0), (2, 2)]
    for pos, lab, prm in zip(positions, labels, params):
        patch = lobe_surface(*prm, size=(190, 165))
        img.paste(patch, pos, patch)
        draw.text((pos[0] + 25, pos[1] + 165), lab, fill=SCU_TEXT, font=F14)
    draw.text((305, 138), "=", fill=SCU_GREEN, font=F22)
    draw.text((555, 138), "+", fill=SCU_GREEN, font=F22)
    draw.text((805, 138), "+", fill=SCU_GREEN, font=F22)
    save(img, "sh_coefficient_projection.png")


def sh_approximation_grid() -> None:
    img, draw = canvas(1500, 980)
    labels = ["Original", "L = 0", "L = 2", "L = 4", "L = 6", "L = 8"]
    rows = [
        [(7, 5), (0, 0), (2, 1), (4, 2), (6, 3), (7, 5)],
        [(6, 4), (0, 0), (2, 2), (4, 1), (5, 3), (6, 4)],
        [(8, 3), (0, 0), (1, 0), (3, 2), (5, 2), (8, 3)],
    ]
    colors = [
        ((50, 205, 70), (35, 150, 90)),
        ((37, 190, 185), (30, 150, 155)),
        ((225, 150, 54), (145, 90, 35)),
    ]
    for i, lab in enumerate(labels):
        x = 35 + i * 242
        draw.text((x + 50, 15), lab, fill=SCU_TEXT, font=F14)
    for row_idx, row in enumerate(rows):
        y = 70 + row_idx * 300
        for i, prm in enumerate(row):
            x = 35 + i * 242
            pos, neg = colors[row_idx]
            patch = lobe_surface(*prm, positive=pos, negative=neg, size=(225, 235))
            img.paste(patch, (x, y), patch)
    save(img, "sh_approximation_grid.png")


def grayscale_signal() -> None:
    img, draw = canvas(900, 620)
    x0, y0, size = 55, 45, 285
    for y in range(size):
        for x in range(size):
            v = int(128 + 72 * math.sin(x / 18) + 42 * math.cos(y / 38) + 35 * math.sin((x + y) / 31))
            v = max(0, min(255, v))
            img.putpixel((x0 + x, y0 + y), (v, v, v))
    draw.rectangle((x0, y0, x0 + size, y0 + size), outline=SCU_GREEN, width=4)
    row = y0 + 178
    pts = []
    for x in range(size):
        px = x0 + x
        v = img.getpixel((px, row))[0]
        pts.append((x, v / 255))
    draw.line((x0, row, x0 + size, row), fill=RED, width=5)
    box = (410, 45, 840, 370)
    plot_axes(draw, box, "x", "I")
    polyline(draw, box, pts, (0, size), (0, 1), fill=SCU_DOT, width=5)
    arrow(draw, (350, row), (405, 300), fill=SCU_GREEN, width=4)
    bar = (410, 455, 840, 490)
    for i in range(bar[0], bar[2] + 1):
        v = int(255 * (i - bar[0]) / max(1, bar[2] - bar[0]))
        draw.line((i, bar[1], i, bar[3]), fill=(v, v, v))
    draw.rectangle(bar, outline=SCU_TEXT, width=2)
    draw.text((bar[0], bar[3] + 10), "0", fill=SCU_TEXT, font=F12)
    draw.text((bar[2] - 42, bar[3] + 10), "255", fill=SCU_TEXT, font=F12)
    save(img, "color_as_signal_grayscale.png")


def directional_distance_signal() -> None:
    img, draw = canvas(760, 610)
    patch = lobe_surface(6, 4, positive=(30, 175, 160), negative=(25, 135, 150), size=(520, 460))
    img.paste(patch, (120, 70), patch)
    cx, cy = 380, 300
    arrow(draw, (cx, cy), (cx + 250, cy), fill=(65, 86, 88), width=4)
    arrow(draw, (cx, cy), (cx - 185, cy + 145), fill=(65, 86, 88), width=4)
    arrow(draw, (cx, cy), (cx, cy - 250), fill=(65, 86, 88), width=4)
    draw.text((cx + 265, cy - 15), "x", fill=SCU_TEXT, font=F14)
    draw.text((cx - 214, cy + 150), "y", fill=SCU_TEXT, font=F14)
    draw.text((cx + 12, cy - 278), "z", fill=SCU_TEXT, font=F14)
    save(img, "directional_distance_signal_custom.png")


def color_mapped_signal() -> None:
    img, draw = canvas(760, 610)
    cx, cy, radius = 380, 300, 235
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            dx, dy = (x - cx) / radius, (y - cy) / radius
            rr = dx * dx + dy * dy
            if rr <= 1:
                z = math.sqrt(1 - rr)
                signal = 0.55 + 0.25 * math.sin(5 * dx + 2 * z) + 0.18 * math.cos(4 * dy - z)
                v = int(max(0, min(255, 255 * signal * (0.70 + 0.30 * max(0, z)))))
                img.putpixel((x, y), (v, v, v))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=SCU_GREEN, width=4)
    save(img, "color_mapped_directional_signal.png")


def multichannel_signal() -> None:
    img, draw = canvas(1150, 720)
    cards = [
        ("R", (230, 75, 70)), ("G", (45, 175, 90)), ("B", (55, 115, 220)),
        ("brightness I", (245, 190, 75)), ("opacity alpha", (100, 130, 150)),
    ]
    positions = [(70, 55), (410, 55), (750, 55), (235, 385), (595, 385)]
    for (label, col), (x, y) in zip(cards, positions):
        draw.rounded_rectangle((x, y, x + 300, y + 240), radius=18, fill=(248, 250, 249), outline=SCU_LIGHT, width=4)
        cx, cy, r = x + 150, y + 105, 70
        for yy in range(cy - r, cy + r + 1):
            for xx in range(cx - r, cx + r + 1):
                dx, dy = (xx - cx) / r, (yy - cy) / r
                if dx * dx + dy * dy <= 1:
                    z = math.sqrt(max(0.0, 1 - dx * dx - dy * dy))
                    signal = 0.58 + 0.25 * math.sin(4 * dx + len(label)) + 0.15 * math.cos(4 * dy)
                    shade = (0.35 + 0.55 * signal) * (0.72 + 0.28 * z)
                    img.putpixel((xx, yy), tuple(max(0, min(255, int(c * shade))) for c in col))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=SCU_GREEN, width=2)
        draw.text((x + 95, y + 195), label, fill=SCU_TEXT, font=F16)
    save(img, "multichannel_directional_signal_custom.png")


def six_sphere_views(name: str, ellipsoid: bool = False) -> None:
    img, draw = canvas(1150, 720)
    labels = ["front", "right", "back", "left", "top", "bottom"]
    positions = [(120, 55), (470, 55), (820, 55), (120, 380), (470, 380), (820, 380)]
    for idx, ((x, y), lab) in enumerate(zip(positions, labels)):
        cx, cy = x + 95, y + 95
        rx, ry = (95, 95) if not ellipsoid else ([120, 80, 110, 78, 95, 105][idx], [70, 115, 78, 108, 55, 130][idx])
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for yy in range(int(cy - ry), int(cy + ry) + 1):
            for xx in range(int(cx - rx), int(cx + rx) + 1):
                dx, dy = (xx - cx) / rx, (yy - cy) / ry
                if dx * dx + dy * dy <= 1:
                    z = math.sqrt(max(0.0, 1 - dx * dx - dy * dy))
                    if ellipsoid:
                        signal = 0.55 + 0.24 * math.sin(3.5 * dx + idx * 0.9) + 0.18 * math.cos(4.0 * dy - idx * 0.6)
                        base = color_mix(signal, lo=(65, 110, 205), mid=(45, 175, 120), hi=(235, 150, 55))
                        shade = 0.65 + 0.35 * z
                        alpha = int(220 * (0.20 + 0.80 * z))
                        col = tuple(int(c * shade) for c in base) + (alpha,)
                    else:
                        signal = 0.55 + 0.28 * math.sin(3 * dx + idx * 0.8) + 0.18 * math.cos(4 * dy - idx)
                        col = color_mix(signal, lo=(45, 85, 180), mid=(40, 180, 110), hi=(235, 160, 55)) + (255,)
                    layer.putpixel((xx, yy), col)
        if ellipsoid:
            layer = layer.filter(ImageFilter.GaussianBlur(8))
        img.paste(layer, (0, 0), layer)
        draw.text((x + 58, y + 210), lab, fill=SCU_TEXT, font=F14)
    save(img, name)


def pipeline_3dgs() -> None:
    img, draw = canvas(1500, 650)
    boxes = [
        (70, 110, 315, 370, "SfM cameras\n+ sparse points"),
        (390, 110, 635, 370, "Initialize\n3D Gaussians"),
        (710, 75, 1035, 405, "Interleaved\noptimization\n+ density control"),
        (1110, 110, 1390, 370, "Visibility-aware\nsplatting\nnovel view"),
    ]
    for x0, y0, x1, y1, label in boxes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=(248, 250, 249), outline=SCU_GREEN, width=4)
        for line_no, line in enumerate(label.split("\n")):
            draw.text((x0 + 26, y0 + 22 + line_no * 34), line, fill=SCU_TEXT, font=F14)
    rng = 0
    for i in range(75):
        rng = (1103515245 * rng + 12345) & 0x7fffffff
        x = 105 + rng % 170
        rng = (1103515245 * rng + 12345) & 0x7fffffff
        y = 210 + rng % 115
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=SCU_DOT)
    for i, (x, y) in enumerate([(455, 240), (500, 280), (555, 225), (575, 305)]):
        draw.ellipse((x - 36, y - 24, x + 36, y + 24), fill=(92, 175, 195), outline=SCU_GREEN, width=3)
    draw.line((470, 270, 575, 225), fill=(190, 190, 190), width=2)
    draw.line((500, 280, 575, 305), fill=(190, 190, 190), width=2)
    draw.rounded_rectangle((775, 195, 970, 255), radius=12, fill=SCU_LIGHT, outline=SCU_GREEN, width=3)
    draw.text((803, 210), "render loss", fill=SCU_TEXT, font=F14)
    draw.rounded_rectangle((775, 285, 970, 345), radius=12, fill=(245, 235, 205), outline=GOLD, width=3)
    draw.text((800, 300), "split / clone", fill=SCU_TEXT, font=F14)
    arrow(draw, (875, 258), (875, 282), fill=SCU_GREEN, width=4)
    arrow(draw, (975, 315), (1015, 255), fill=SCU_GREEN, width=4)
    for y in range(225, 325):
        for x in range(1180, 1320):
            t = (x - 1180) / 140
            s = (y - 225) / 100
            col = color_mix(0.35 + 0.45 * math.sin(5 * t) * math.cos(3 * s), lo=(50, 90, 180), mid=(50, 180, 110), hi=(235, 160, 60))
            img.putpixel((x, y), col)
    draw.rectangle((1180, 225, 1320, 325), outline=SCU_GREEN, width=3)
    for start, end in [((315, 240), (390, 240)), ((635, 240), (710, 240)), ((1035, 240), (1110, 240))]:
        arrow(draw, start, end, fill=SCU_GREEN, width=5)
    save(img, "3dgs_pipeline_redrawn.png")


def blank_3dgs_examples() -> None:
    for name in ["trained_gaussians_placeholder_a.png", "trained_gaussians_placeholder_b.png"]:
        img, draw = canvas(760, 520)
        draw.rectangle((22, 22, 738, 498), outline=(205, 214, 212), width=3)
        save(img, name, pad=10)


def main() -> None:
    square_wave_approximation()
    polar_dumbbell()
    two_variable_surface()
    two_variable_basis_functions()
    two_variable_fit_grid()
    sh_basis_grid()
    sh_coefficient_projection()
    sh_approximation_grid()
    grayscale_signal()
    directional_distance_signal()
    color_mapped_signal()
    multichannel_signal()
    six_sphere_views("sh_color_six_views.png", ellipsoid=False)
    six_sphere_views("gaussian_ellipsoid_six_views.png", ellipsoid=True)
    pipeline_3dgs()
    blank_3dgs_examples()


if __name__ == "__main__":
    main()
