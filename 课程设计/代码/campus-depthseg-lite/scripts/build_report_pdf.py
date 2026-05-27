"""Build a more readable LaTeX-style course report PDF.

The script writes a LaTeX source file and a PDF draft.  The PDF is rendered
with ReportLab because a local TeX compiler may not be available on Windows.
The generated LaTeX source keeps the report editable for later manual use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "课程设计报告_LaTeX风格版.pdf"
LATEX_DIR = PROJECT_ROOT / "report_latex"
LATEX_PATH = LATEX_DIR / "main.tex"
PDF_ASSET_DIR = PROJECT_ROOT / "outputs" / "report_pdf_assets"


@dataclass(frozen=True)
class FigureSpec:
    number: str
    title: str
    path: str
    landscape_page: bool = False
    width_ratio: float = 0.95
    height_ratio: float = 0.78


RESULT_ROWS = [
    ["RGB-only", "RGB", "无", "0.4226", "0.5964", "0.5881", "1.5588"],
    ["RGBD-concat", "RGB + Depth", "直接拼接", "0.4683", "0.6318", "0.6294", "1.4205"],
    ["RGBD-boundary", "RGB + Depth", "深度边界残差", "0.4206", "0.5974", "0.5925", "1.5317"],
    ["RGBD-concat-boundary", "RGB + Depth", "拼接 + 边界残差", "0.4605", "0.6271", "0.6191", "1.4378"],
]

IOU_ROWS = [
    ["RGB-only", "0.3549", "0.4731", "0.5495", "0.4239", "0.3114"],
    ["RGBD-concat", "0.3857", "0.5751", "0.5647", "0.4722", "0.3437"],
    ["RGBD-boundary", "0.3476", "0.4629", "0.5443", "0.4276", "0.3205"],
    ["RGBD-concat-boundary", "0.3636", "0.5530", "0.5595", "0.4899", "0.3366"],
]

MODEL_ROWS = [
    ["rgb", "1,511,689", "1.5117 M"],
    ["rgbd_concat", "1,512,121", "1.5121 M"],
    ["rgbd_boundary", "1,512,413", "1.5124 M"],
    ["rgbd_concat_boundary", "1,512,845", "1.5128 M"],
]


def main() -> None:
    PDF_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    assets = build_report_friendly_assets()
    write_latex(assets)
    build_pdf(assets)
    print(f"latex: {LATEX_PATH}")
    print(f"pdf: {PDF_PATH}")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc"))
    pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
    pdfmetrics.registerFont(TTFont("KaiTi", "C:/Windows/Fonts/simkai.ttf"))


def build_report_friendly_assets() -> dict[str, str]:
    assets = {
        "class_dist": "outputs/figures/nyu5_class_distribution.png",
        "method_compare": "outputs/report_assets/method_comparison/method_comparison_grid.png",
        "confusion": "outputs/report_assets/confusion_matrix/confusion_matrix_normalized.png",
        "best_cases": "outputs/report_assets/error_cases/best_cases.png",
    }
    crop_vertical(
        PROJECT_ROOT / "outputs/figures/nyu5_gallery_train.png",
        PDF_ASSET_DIR / "nyu5_gallery_train_readable.png",
        0.00,
        0.43,
    )
    crop_vertical(
        PROJECT_ROOT / "outputs/figures/nyu5_gallery_val.png",
        PDF_ASSET_DIR / "nyu5_gallery_val_readable.png",
        0.00,
        0.43,
    )
    crop_vertical(
        PROJECT_ROOT / "outputs/figures/nyu5_gallery_test.png",
        PDF_ASSET_DIR / "nyu5_gallery_test_readable.png",
        0.00,
        0.43,
    )
    crop_vertical(
        PROJECT_ROOT / "outputs/report_assets/campus_demo/campus_demo_gallery.png",
        PDF_ASSET_DIR / "campus_demo_gallery_top.png",
        0.00,
        0.52,
    )
    crop_vertical(
        PROJECT_ROOT / "outputs/report_assets/campus_demo/campus_demo_gallery.png",
        PDF_ASSET_DIR / "campus_demo_gallery_bottom.png",
        0.48,
        1.00,
    )
    crop_vertical(
        PROJECT_ROOT / "outputs/report_assets/method_comparison/method_comparison_grid.png",
        PDF_ASSET_DIR / "method_comparison_top.png",
        0.00,
        0.52,
    )
    crop_vertical(
        PROJECT_ROOT / "outputs/report_assets/method_comparison/method_comparison_grid.png",
        PDF_ASSET_DIR / "method_comparison_bottom.png",
        0.48,
        1.00,
    )
    make_architecture_diagram(PDF_ASSET_DIR / "campus_depthseg_lite_architecture.png")
    make_curve_panel(
        [
            PROJECT_ROOT / "outputs/runs/exp02_rgbd_concat_e20/training_loss_curve.png",
            PROJECT_ROOT / "outputs/runs/exp02_rgbd_concat_e20/validation_miou_curve.png",
            PROJECT_ROOT / "outputs/runs/exp02_rgbd_concat_e20/validation_pixel_acc_curve.png",
        ],
        PDF_ASSET_DIR / "rgbd_concat_training_curves_panel.png",
    )
    assets.update(
        {
            "train_gallery": "outputs/report_pdf_assets/nyu5_gallery_train_readable.png",
            "val_gallery": "outputs/report_pdf_assets/nyu5_gallery_val_readable.png",
            "test_gallery": "outputs/report_pdf_assets/nyu5_gallery_test_readable.png",
            "curves": "outputs/report_pdf_assets/rgbd_concat_training_curves_panel.png",
            "architecture": "outputs/report_pdf_assets/campus_depthseg_lite_architecture.png",
            "method_top": "outputs/report_pdf_assets/method_comparison_top.png",
            "method_bottom": "outputs/report_pdf_assets/method_comparison_bottom.png",
        }
    )
    campus_panels = sorted(
        path
        for path in (PROJECT_ROOT / "outputs/report_assets/campus_demo").glob("campus_demo_*.png")
        if path.stem.removeprefix("campus_demo_").isdigit()
    )
    if not campus_panels:
        raise FileNotFoundError("Missing campus demo panels under outputs/report_assets/campus_demo")
    for idx, panel_path in enumerate(campus_panels, start=1):
        assets[f"campus_{idx:03d}"] = str(panel_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    missing = [path for path in assets.values() if not (PROJECT_ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing report assets: {missing}")
    return assets


def crop_vertical(src: Path, dst: Path, start: float, end: float) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing image: {src}")
    image = PILImage.open(src).convert("RGB")
    width, height = image.size
    top = int(height * start)
    bottom = int(height * end)
    image.crop((0, top, width, bottom)).save(dst)


def make_curve_panel(paths: list[Path], dst: Path) -> None:
    images = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        images.append(PILImage.open(path).convert("RGB"))
    target_w = 900
    resized = []
    for image in images:
        new_h = int(image.height * target_w / image.width)
        resized.append(image.resize((target_w, new_h), PILImage.Resampling.LANCZOS))
    gap = 40
    canvas = PILImage.new("RGB", (target_w, sum(img.height for img in resized) + gap * 2), "white")
    y = 0
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height + gap
    canvas.save(dst)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_path = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    return ImageFont.truetype(font_path, size=size)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    line_spacing: int = 8,
) -> None:
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_spacing * (len(lines) - 1)
    y = box[1] + ((box[3] - box[1]) - total_h) // 2
    for line, width, height in zip(lines, widths, heights):
        x = box[0] + ((box[2] - box[0]) - width) // 2
        draw.text((x, y), line, font=font, fill=fill)
        y += height + line_spacing


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=3)
    draw_centered_text(draw, box, title, font, outline)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) >= abs(dy):
        direction = 1 if dx >= 0 else -1
        points = [(end[0], end[1]), (end[0] - 18 * direction, end[1] - 10), (end[0] - 18 * direction, end[1] + 10)]
    else:
        direction = 1 if dy >= 0 else -1
        points = [(end[0], end[1]), (end[0] - 10, end[1] - 18 * direction), (end[0] + 10, end[1] - 18 * direction)]
    draw.polygon(points, fill=color)


def draw_elbow_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int = 5,
) -> None:
    if len(points) < 2:
        raise ValueError("Elbow arrow needs at least two points")
    if len(points) > 2:
        draw.line(points[:-1], fill=color, width=width)
    draw_arrow(draw, points[-2], points[-1], color, width=width)


def make_architecture_diagram(dst: Path) -> None:
    width, height = 2200, 1350
    image = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(52, bold=True)
    subtitle_font = load_font(28)
    box_font = load_font(30, bold=True)
    mid_font = load_font(24, bold=True)
    small_font = load_font(22)
    tiny_font = load_font(18)

    blue = (29, 101, 204)
    teal = (0, 130, 130)
    orange = (235, 126, 38)
    purple = (112, 69, 181)
    red = (210, 55, 70)
    gray = (85, 92, 102)

    draw_centered_text(draw, (0, 35, width, 105), "CampusDepthSegLite RGB-D Segmentation Framework", title_font, (20, 20, 20))
    draw_centered_text(
        draw,
        (0, 110, width, 155),
        "Lightweight five-class indoor segmentation with occupancy and risk visualization",
        subtitle_font,
        (90, 90, 90),
    )

    def column_label(text: str, x0: int, x1: int, color: tuple[int, int, int]) -> None:
        draw_centered_text(draw, (x0, 178, x1, 218), text, mid_font, color)
        draw.line((x0 + 15, 224, x1 - 15, 224), fill=color, width=4)

    column_label("Inputs", 60, 320, gray)
    column_label("Input Mode", 370, 670, orange)
    column_label("Mini Encoder", 720, 1080, blue)
    column_label("Depth Cue", 1130, 1480, orange)
    column_label("Decoder", 1530, 1805, purple)
    column_label("Outputs", 1870, 2140, red)

    rgb_box = (70, 310, 320, 470)
    depth_box = (70, 760, 320, 920)
    draw_box(draw, rgb_box, "RGB image\n3 channels", (232, 242, 255), blue, box_font)
    draw_box(draw, depth_box, "Depth map\n1 channel", (230, 250, 248), teal, box_font)

    mode_box = (380, 320, 670, 940)
    draw.rounded_rectangle(mode_box, radius=26, fill=(255, 250, 238), outline=orange, width=4)
    draw_centered_text(draw, (395, 350, 655, 430), "Variant\nconstruction", box_font, orange)
    mode_items = [
        ("rgb", "RGB only"),
        ("rgbd_concat", "RGB + depth"),
        ("rgbd_boundary", "RGB + Sobel edge"),
        ("concat_boundary", "RGB + depth + edge"),
    ]
    for i, (name, note) in enumerate(mode_items):
        y = 480 + i * 90
        draw.rounded_rectangle((420, y, 630, y + 52), radius=12, fill=(255, 255, 255), outline=orange, width=2)
        draw.text((438, y + 8), name, font=tiny_font, fill=orange)
        draw.text((438, y + 30), note, font=tiny_font, fill=gray)

    encoder_box = (730, 270, 1080, 990)
    draw.rounded_rectangle(encoder_box, radius=28, fill=(240, 247, 255), outline=blue, width=4)
    draw_centered_text(draw, (750, 300, 1060, 370), "MiniHierarchicalEncoder", mid_font, blue)
    stage_info = [
        ("Stage 1", "48 ch", "1/2"),
        ("Stage 2", "96 ch", "1/4"),
        ("Stage 3", "192 ch", "1/8"),
        ("Stage 4", "384 ch", "1/16"),
    ]
    for i, (stage, ch, scale) in enumerate(stage_info):
        y = 420 + i * 125
        draw.rounded_rectangle((770, y, 1040, y + 76), radius=14, fill=(219, 235, 255), outline=blue, width=2)
        draw.text((790, y + 12), stage, font=small_font, fill=blue)
        draw.text((925, y + 12), ch, font=small_font, fill=gray)
        draw.text((925, y + 43), scale, font=tiny_font, fill=gray)
        if i < 3:
            draw_arrow(draw, (905, y + 78), (905, y + 116), blue, width=3)
    draw.text((772, 930), "Depthwise conv + pointwise conv + residual", font=tiny_font, fill=gray)
    draw.text((772, 960), "Four scales are sent to the decoder", font=tiny_font, fill=gray)

    fusion_box = (1130, 320, 1480, 940)
    draw.rounded_rectangle(fusion_box, radius=28, fill=(255, 247, 237), outline=orange, width=4)
    draw_centered_text(draw, (1150, 350, 1460, 430), "DepthBoundary\nResidualFusion", mid_font, orange)
    draw.rounded_rectangle((1175, 480, 1435, 552), radius=12, fill=(255, 255, 255), outline=orange, width=2)
    draw_centered_text(draw, (1175, 480, 1435, 552), "Fixed Sobel edge", small_font, orange)
    draw.rounded_rectangle((1175, 620, 1435, 725), radius=12, fill=(255, 255, 255), outline=orange, width=2)
    draw_centered_text(draw, (1175, 620, 1435, 725), "F_i' = F_i\n+ alpha_i * edge_i", tiny_font, gray)
    draw.text((1175, 820), "Used by boundary variants only", font=tiny_font, fill=gray)

    decoder_box = (1530, 320, 1805, 940)
    draw.rounded_rectangle(decoder_box, radius=28, fill=(245, 239, 255), outline=purple, width=4)
    draw_centered_text(draw, (1550, 350, 1785, 430), "Weighted-FPN\nDecoder", mid_font, purple)
    for i, label in enumerate(["1x1 lateral", "Upsample", "Softplus weights", "3x3 smooth", "Classifier"]):
        y = 480 + i * 72
        draw.rounded_rectangle((1570, y, 1765, y + 44), radius=10, fill=(255, 255, 255), outline=purple, width=2)
        draw_centered_text(draw, (1570, y, 1765, y + 44), label, tiny_font, purple)

    logits_box = (1880, 300, 2140, 440)
    mask_box = (1880, 600, 2140, 740)
    analysis_box = (1880, 900, 2140, 1040)
    draw_box(draw, logits_box, "5-class logits\n[B,5,H,W]", (255, 244, 246), red, mid_font)
    draw_box(draw, mask_box, "Segmentation\nmask", (255, 244, 246), red, mid_font)
    draw_box(draw, analysis_box, "Inspection\noccupancy + risk", (255, 246, 247), red, mid_font)

    draw_arrow(draw, (320, 390), (380, 520), blue)
    draw_arrow(draw, (320, 840), (380, 700), teal)
    draw_arrow(draw, (670, 630), (730, 630), orange)
    draw_arrow(draw, (1080, 630), (1130, 630), blue)
    draw_elbow_arrow(draw, [(320, 840), (350, 1080), (1120, 1080), (1120, 516), (1175, 516)], teal, width=4)
    draw_arrow(draw, (1480, 630), (1530, 630), orange)
    draw_arrow(draw, (1805, 630), (1880, 370), purple)
    draw_arrow(draw, (2010, 440), (2010, 600), red)
    draw_arrow(draw, (2010, 740), (2010, 900), red)

    legend = (70, 1130, 1760, 1240)
    draw.rounded_rectangle(legend, radius=18, fill=(250, 250, 250), outline=(165, 165, 165), width=2)
    legend_items = [
        (blue, "RGB features"),
        (teal, "Depth / edge"),
        (orange, "Fusion"),
        (purple, "Decoder"),
        (red, "Output"),
    ]
    for i, (color, text) in enumerate(legend_items):
        x = 105 + i * 325
        draw.rounded_rectangle((x, 1168, x + 46, 1190), radius=5, fill=color, outline=color)
        draw.text((x + 60, 1162), text, font=tiny_font, fill=(50, 50, 50))

    draw_centered_text(
        draw,
        (0, 1270, width, 1325),
        "Clean-room course-project diagram: simple CNN encoder, optional depth-edge residual fusion, and weighted FPN decoding.",
        small_font,
        (110, 110, 110),
    )
    image.save(dst)


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="SimHei",
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=18,
            wordWrap="CJK",
        ),
        "cover": ParagraphStyle(
            "cover",
            parent=base["Normal"],
            fontName="SimSun",
            fontSize=13,
            leading=24,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="SimHei",
            fontSize=16,
            leading=23,
            spaceBefore=14,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="SimHei",
            fontSize=14,
            leading=21,
            spaceBefore=10,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="SimSun",
            fontSize=11.5,
            leading=20,
            firstLineIndent=23,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body_no_indent": ParagraphStyle(
            "body_no_indent",
            parent=base["Normal"],
            fontName="SimSun",
            fontSize=11.5,
            leading=20,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="KaiTi",
            fontSize=10.5,
            leading=16,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "table_caption": ParagraphStyle(
            "table_caption",
            parent=base["Normal"],
            fontName="KaiTi",
            fontSize=10.5,
            leading=16,
            alignment=TA_CENTER,
            spaceBefore=8,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "ref": ParagraphStyle(
            "ref",
            parent=base["Normal"],
            fontName="SimSun",
            fontSize=9.5,
            leading=15,
            leftIndent=14,
            firstLineIndent=-14,
            wordWrap="CJK",
        ),
    }


def build_pdf(assets: dict[str, str]) -> None:
    register_fonts()
    styles = build_styles()
    portrait_page = A4
    landscape_page = landscape(A4)
    margin = 2 * cm
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=portrait_page, leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin)
    portrait_frame = Frame(margin, margin, portrait_page[0] - 2 * margin, portrait_page[1] - 2 * margin, id="portrait")
    landscape_frame = Frame(margin, margin, landscape_page[0] - 2 * margin, landscape_page[1] - 2 * margin, id="landscape")
    doc.addPageTemplates(
        [
            PageTemplate(id="portrait", frames=[portrait_frame], onPage=draw_page_number),
            PageTemplate(id="landscape", pagesize=landscape_page, frames=[landscape_frame], onPage=draw_page_number),
        ]
    )

    story: list = []
    add_cover(story, styles)
    add_abstract(story, styles)
    add_section_1(story, styles)
    add_section_2(story, styles)
    add_section_3(story, styles, assets)
    add_section_4(story, styles, assets)
    add_section_5(story, styles, assets)
    add_section_6(story, styles)
    add_references(story, styles)
    add_appendices(story, styles)
    doc.build(story)


def draw_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("SimSun", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(doc.pagesize[0] / 2, 1.1 * cm, str(doc.page))
    canvas.restoreState()


def para(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def heading(text: str, styles: dict[str, ParagraphStyle], level: int = 1) -> Paragraph:
    return Paragraph(text, styles["h1" if level == 1 else "h2"])


def add_cover(story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Spacer(1, 2.5 * cm))
    story.append(Paragraph("湖北大学本科课程设计", styles["title"]))
    story.append(Spacer(1, 1.4 * cm))
    for line in [
        "课程名称：《人工智能技术与应用》课程设计",
        "课设题目：面向校园室内空间巡检的轻量 RGB-D 语义分割与空间占用分析系统",
        "姓名：易唯　　学号：202231116020106",
        "专业年级：计算机科学与技术2022级",
        "指导教师/职称：王雷春/副教授",
        "日期：2026年6月",
    ]:
        story.append(Paragraph(line, styles["cover"]))
        story.append(Spacer(1, 0.25 * cm))
    story.append(PageBreak())


def add_abstract(story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(heading("摘 要", styles, 1))
    story.append(
        para(
            "本课程设计关注的是一个很具体的校园场景：教室、实验室、走廊或图书馆里，经常会出现桌椅、纸箱、书包等物品临时占用通道的情况。人工巡检可以发现这些问题，但记录方式比较主观，也不容易形成稳定的数据结果。因此，本项目尝试用语义分割方法，把一张室内图像划分成地面、墙面、障碍物、门窗和其他区域，再进一步给出空间占用比例、障碍区域框和简单风险提示。",
            styles,
        )
    )
    story.append(
        para(
            "项目使用 NYUDepthV2 数据集导出 1449 张 RGB-D 样本，并将原始标签重新整理成 other、floor、wall、obstacle、door_window 五类。模型部分实现了一个轻量的 CampusDepthSegLite，包含四阶段编码器和 Weighted-FPN 解码器。为了比较 RGB 与 depth 的作用，实验设计了 RGB-only、RGBD-concat、RGBD-boundary、RGBD-concat-boundary 四个变体。测试结果显示，RGBD-concat 的整体效果最好，mIoU 为 0.4683；相比 RGB-only 提升 4.57 个百分点，说明完整 depth 信息对室内空间分割有帮助。RGBD-concat-boundary 的 obstacle IoU 最高，达到 0.4899，说明深度边界对障碍区域有一定帮助，但它的整体 mIoU 没有超过 RGBD-concat。",
            styles,
        )
    )
    story.append(
        para(
            "最后，本项目还使用自采集校园 RGB 图像做了定性演示。由于这些图片没有真实 depth 和像素级标注，所以它们不参与训练，也不计算 mIoU、Pixel Acc 等定量指标，只用于展示系统从图像输入到预测 mask、占用图、风险框和文字摘要的完整流程。",
            styles,
        )
    )
    story.append(Paragraph("关键词：RGB-D 语义分割；校园巡检；空间占用分析；轻量模型；卷积神经网络", styles["body_no_indent"]))
    story.append(PageBreak())
    story.append(heading("目录", styles, 1))
    story.append(para("本 PDF 版为课程设计初稿，目录可在 Word 或 LaTeX 编译环境中自动更新。", styles, "body_no_indent"))
    story.append(PageBreak())


def add_section_1(story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(heading("1 引言", styles, 1))
    story.append(heading("1.1 选题背景", styles, 2))
    story.append(para("这次课程设计的想法来自校园里比较常见的室内空间使用问题。比如实验室门口堆放箱子、走廊旁边临时放置桌椅、图书馆或自习区附近有物品占用通道等。这些情况通常不一定严重到需要报警，但如果长期存在，就会影响通行和空间管理。人工巡检当然可以处理这些问题，不过人工记录往往比较零散，也不容易把“地面可见多少”“障碍物大概占了多少区域”这类信息量化下来。", styles))
    story.append(para("在《人工智能技术与应用》课程的导论部分，课件强调了人工智能应用需要从实际场景出发，同时也要认识到技术边界。本项目正是一个偏应用型的小系统：它不是要替代人工巡检，而是尝试把计算机视觉结果转化为更直观的辅助信息。", styles))
    story.append(heading("1.2 任务目标", styles, 2))
    story.append(para("本项目的核心任务是：输入一张室内 RGB 或 RGB-D 图像，输出五类语义分割结果，并根据分割 mask 计算空间占用指标。五类区域分别是 other、floor、wall、obstacle 和 door_window。这样的划分比原始 NYUDepthV2 类别更粗，但更贴近校园巡检：巡检人员更关心地面是否被遮挡、障碍物在哪里、门窗区域是否明显，而不是把每一种家具都分成很细的类别。", styles))
    story.append(heading("1.3 本文主要工作", styles, 2))
    story.append(para("本文主要完成了四件事。第一，整理 NYUDepthV2 数据，把原始室内类别映射成五类校园空间标签。第二，设计并实现轻量分割模型 CampusDepthSegLite。第三，完成四种 RGB-D 融合方式的消融实验。第四，在模型输出基础上实现空间占用分析和自采集校园场景的定性展示。整体思路尽量保持简单、清楚、能运行，而不是堆叠复杂结构。", styles))


def add_section_2(story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(heading("2 相关理论与课程知识对应", styles, 1))
    story.append(heading("2.1 机器学习与深度学习流程", styles, 2))
    story.append(para("课程中机器学习基础部分讲到，一个完整的学习系统通常包括数据、模型、损失函数、优化方法和评价指标。本项目也按这个流程展开：先从 NYUDepthV2 导出 RGB、depth 和 label，再设计模型，使用 CrossEntropyLoss 与 DiceLoss 训练，最后用 mIoU、Pixel Acc、Mean Acc 和 per-class IoU 评价。", styles))
    story.append(heading("2.2 卷积神经网络与图像特征", styles, 2))
    story.append(para("卷积神经网络课件中介绍了卷积的局部感受野和权重共享思想。语义分割任务需要保留空间位置，因此卷积网络很适合处理这类问题。本项目没有使用很大的预训练模型，而是实现了一个四阶段轻量 CNN 编码器，每个阶段逐步降低分辨率、增加通道数，用较少参数提取多尺度特征。", styles))
    story.append(heading("2.3 RGB-D 信息为什么有用", styles, 2))
    story.append(para("RGB 图像可以看到颜色和纹理，但室内场景里很多区域颜色相近，例如白墙、浅色地面、木质桌面，在 RGB 上容易混淆。Depth 提供的是相对距离和空间结构，可以帮助模型理解哪里更像地面、哪里更像墙面或障碍物。因此，本项目专门比较 RGB-only 和 RGBD-concat，观察 depth 是否真的带来提升。", styles))
    story.append(heading("2.4 评价指标", styles, 2))
    story.append(para("语义分割不能只看整体准确率，因为类别面积差异很大。比如墙面和地面像素很多，而 door_window 面积较小。mIoU 会分别计算每个类别的交并比再求平均，因此更能反映模型是否兼顾了不同类别。本文把 mIoU 作为主要指标，同时保留 Pixel Acc、Mean Acc 和 per-class IoU 辅助分析。", styles))


def add_section_3(story: list, styles: dict[str, ParagraphStyle], assets: dict[str, str]) -> None:
    story.append(heading("3 数据集与预处理", styles, 1))
    story.append(heading("3.1 NYUDepthV2 数据", styles, 2))
    story.append(para("本项目使用 NYUDepthV2 数据集，从 nyu_depth_v2_labeled.mat 中导出 1449 张 RGB-D 样本。划分比例约为 7:1.5:1.5，最终训练集 1014 张、验证集 217 张、测试集 218 张。所有输入统一 resize 到 240 x 320。", styles))
    story.append(heading("3.2 五类标签映射", styles, 2))
    story.append(para("原始 NYUDepthV2 的标签类别比较多，如果全部用于课程设计，模型和分析都会变得复杂。为了贴近校园巡检，本项目将其合并为五类：other 表示其他区域，floor 表示地面，wall 表示墙面或天花板，obstacle 表示桌椅、柜子、箱包等可能影响通行的物体，door_window 表示门窗及相关区域。", styles))
    story.append(make_table("表3-1 数据集划分", ["Split", "Samples"], [["train", "1014"], ["val", "217"], ["test", "218"]], styles))
    add_landscape_figure(story, assets["class_dist"], "图3-1 NYU5 五类标签像素分布", styles)
    add_portrait_figure(story, assets["train_gallery"], "图3-2 NYUDepthV2 样例可视化（报告放大版）", styles)


def add_section_4(story: list, styles: dict[str, ParagraphStyle], assets: dict[str, str]) -> None:
    story.append(heading("4 模型设计与系统实现", styles, 1))
    story.append(heading("4.1 总体流程", styles, 2))
    story.append(para("系统分为数据准备、模型预测、空间占用分析和可视化输出四个部分。训练和测试阶段使用 NYUDepthV2 的 RGB-D 数据；自采集校园图像只用于定性展示。模型输出五类 mask 后，后处理模块会计算 floor_ratio、lower_floor_ratio、obstacle_ratio 和 risk_score，并把 obstacle 区域通过连通域分析画成红色区域框。", styles))
    story.append(heading("4.2 CampusDepthSegLite", styles, 2))
    story.append(para("CampusDepthSegLite 的设计目标是轻量和可读。编码器通道为 [48, 96, 192, 384]，每个 stage 使用下采样卷积和轻量残差块；解码器使用 128 通道 Weighted-FPN，把四个尺度的特征统一后加权融合，再上采样回原图大小。这样做的好处是结构清楚，参数量约 1.51M，也方便做消融实验。", styles))
    story.append(heading("4.3 四种融合方式", styles, 2))
    story.append(para("四种变体的区别比较直接：RGB-only 只输入 RGB；RGBD-concat 把 RGB 和 depth 拼成四通道输入；RGBD-boundary 不直接拼接完整 depth，而是提取 depth 的 Sobel 边界后注入多尺度特征；RGBD-concat-boundary 同时使用完整 depth 和边界信息。这样的设置可以回答两个问题：完整 depth 有没有帮助？边界信息能不能进一步补充障碍物分割？", styles))
    story.append(make_table("表4-1 模型复杂度统计", ["Variant", "Params", "Params (M)"], MODEL_ROWS, styles))
    add_landscape_figure(story, assets["architecture"], "图4-1 CampusDepthSegLite 系统结构示意图", styles)


def add_section_5(story: list, styles: dict[str, ParagraphStyle], assets: dict[str, str]) -> None:
    story.append(heading("5 实验结果与分析", styles, 1))
    story.append(heading("5.1 训练过程", styles, 2))
    story.append(para("四个实验都训练 20 epoch，优化器为 AdamW，学习率为 1e-4，batch size 为 4，checkpoint 按验证集 val_mIoU 保存。下面以整体表现最好的 RGBD-concat 为例展示训练曲线。可以看到训练损失整体下降，验证 mIoU 和 Pixel Acc 前期提升较明显，后期趋于稳定。", styles))
    story.append(para("课程中讲机器学习流程时反复强调，训练集、验证集和测试集要分开使用，不能只看训练损失来判断模型好坏。因此这里保存 best checkpoint 时没有看测试集，而是只监控验证集 val_mIoU；测试集只在训练结束后用于最终评价。这样的流程虽然比直接挑测试集最高分麻烦一点，但结果更可信。", styles))
    story.append(para("从曲线的形状看，训练损失没有出现明显发散，验证指标也不是一直停在零附近，说明数据读取、标签映射、loss 和 metric 的链路是连通的。不过验证 mIoU 后期有轻微波动，这也提醒我们：这个模型是课程设计里的轻量版本，不能把它写成工程部署级系统，报告里只把它作为一个可运行的校园巡检原型来分析。", styles))
    add_portrait_figure(story, assets["curves"], "图5-1 RGBD-concat 训练曲线汇总", styles)
    story.append(heading("5.2 主实验结果", styles, 2))
    story.append(para("从表5-1可以看出，RGBD-concat 是四个变体中整体最好的模型，test mIoU 为 0.4683。相比 RGB-only 的 0.4226，提升了 4.57 个百分点。这说明在室内空间分割任务中，完整 depth 信息确实能补充 RGB 图像中不明显的几何结构。", styles))
    story.append(make_table("表5-1 四种模型变体测试集结果", ["Method", "Input", "Fusion", "mIoU", "Pixel Acc", "Mean Acc", "Test Loss"], RESULT_ROWS, styles))
    story.append(heading("5.3 分类别 IoU", styles, 2))
    story.append(para("per-class IoU 能看到更细的变化。RGBD-concat 在 floor、wall 和 door_window 上表现较好；RGBD-concat-boundary 的 obstacle IoU 最高，为 0.4899。这说明边界信息对障碍物区域确实有帮助，但它没有让整体 mIoU 超过 RGBD-concat，所以报告中不能把 boundary fusion 写成整体最佳。", styles))
    story.append(make_table("表5-2 四种模型变体 per-class IoU", ["Method", "other", "floor", "wall", "obstacle", "door_window"], IOU_ROWS, styles))
    add_landscape_figure(story, assets["method_top"], "图5-2 四种方法预测对比（一，报告放大版）", styles)
    add_landscape_figure(story, assets["method_bottom"], "图5-3 四种方法预测对比（二，报告放大版）", styles)
    story.append(heading("5.4 混淆矩阵与可视化分析", styles, 2))
    story.append(para("混淆矩阵显示，door_window 容易和 wall、other 混淆。这也符合直观观察：门窗区域有时面积小、边缘不完整，而且在室内图像里常被家具或光照影响。为了避免报告过长，这里保留归一化混淆矩阵和表现较好的测试样例，不再单独展示较差样例。", styles))
    add_portrait_figure(story, assets["confusion"], "图5-4 最佳模型归一化混淆矩阵", styles)
    add_portrait_figure(story, assets["best_cases"], "图5-5 测试集中表现较好的样例", styles)
    story.append(heading("5.5 自采集校园场景定性展示", styles, 2))
    story.append(para("自采集图像来自图书馆、实验室、教室、走廊和遮挡较多的室内场景。这部分图片没有像素级 GT，也没有真实 depth，因此不参与训练、验证或测试，也不计算 mIoU、Pixel Acc、Mean Acc。这里使用 RGB-only checkpoint 进行推理，只是为了展示系统在真实校园照片上的可视化流程，包括 Prediction mask、Occupancy map、Risk boxes、Metrics block 和 Text summary。", styles))
    campus_indices = [int(key.removeprefix("campus_")) for key in sorted(assets) if key.startswith("campus_")]
    for start in range(1, len(campus_indices) + 1, 2):
        figures = [(assets[f"campus_{start:03d}"], f"图5-{5 + start} 自采集校园场景 {start:03d}")]
        if start + 1 <= len(campus_indices):
            figures.append((assets[f"campus_{start + 1:03d}"], f"图5-{6 + start} 自采集校园场景 {start + 1:03d}"))
        add_two_figures_page(
            story,
            figures,
            styles,
        )


def add_section_6(story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(heading("6 总结与展望", styles, 1))
    story.append(heading("6.1 工作总结", styles, 2))
    story.append(para("本课程设计完成了从数据准备、模型设计、训练评估到可视化演示的一条完整链路。实验结果比较明确：完整 depth 直接拼接是当前最稳的方案；深度边界对 obstacle 有帮助，但整体没有超过 RGBD-concat。系统后处理部分把分割结果转成空间占用指标和风险提示，使模型输出更接近校园巡检应用。", styles))
    story.append(heading("6.2 不足与局限性", styles, 2))
    story.append(para("本项目仍然有明显局限。第一，NYUDepthV2 与真实校园图像存在 domain gap。第二，自采集图片没有真实 depth 和像素级标注，不能作为定量泛化证明。第三，risk_score 是启发式规则，只能作为提示，不能直接用于真实安全决策。第四，door_window 类仍然较难分割，后续需要更多数据或更合适的融合策略。", styles))
    story.append(heading("6.3 后续改进", styles, 2))
    story.append(para("后续可以采集带真实 depth 或人工标注的校园数据，进一步改进 depth 融合方式；也可以把空间占用规则做得更细，例如区分走廊、门口和桌面遮挡。真正用于巡检时，还需要人工复核和隐私保护机制，不能只依赖模型自动判断。", styles))


def add_references(story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(PageBreak())
    story.append(heading("参考文献", styles, 1))
    refs = [
        "[1] Silberman N, Hoiem D, Kohli P, et al. Indoor segmentation and support inference from RGBD images[C]//European Conference on Computer Vision. 2012: 746-760.",
        "[2] Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3431-3440.",
        "[3] Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation[C]//MICCAI. 2015: 234-241.",
        "[4] Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//ECCV. 2018: 801-818.",
        "[5] Lin T Y, Dollar P, Girshick R, et al. Feature pyramid networks for object detection[C]//CVPR. 2017: 2117-2125.",
        "[6] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//CVPR. 2016: 770-778.",
        "[7] Badrinarayanan V, Kendall A, Cipolla R. SegNet: a deep convolutional encoder-decoder architecture for image segmentation[J]. IEEE TPAMI, 2017, 39(12): 2481-2495.",
        "[8] Hazirbas C, Ma L, Domokos C, et al. FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture[C]//ACCV. 2016: 213-228.",
        "[9] Seichter D, Kohler M, Lewandowski B, et al. Efficient RGB-D semantic segmentation for indoor scene analysis[C]//ICRA. 2021: 13525-13531.",
        "[10] Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library[C]//NeurIPS. 2019: 8024-8035.",
        "[11] Kingma D P, Ba J. Adam: a method for stochastic optimization[C]//ICLR. 2015.",
        "[12] Milletari F, Navab N, Ahmadi S A. V-Net: fully convolutional neural networks for volumetric medical image segmentation[C]//3DV. 2016: 565-571.",
        "[13] 王雷春. 第1章 人工智能导论[Z]. 《人工智能技术与应用》课程课件, 2026.",
        "[14] 王雷春. 第2章 机器学习基础[Z]. 《人工智能技术与应用》课程课件, 2026.",
        "[15] 王雷春. 第3章 神经网络基础[Z]. 《人工智能技术与应用》课程课件, 2026.",
        "[16] 王雷春. 第4章 卷积神经网络及应用[Z]. 《人工智能技术与应用》课程课件, 2026.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles["ref"]))


def add_appendices(story: list, styles: dict[str, ParagraphStyle]) -> None:
    story.append(heading("附录A 核心代码说明", styles, 1))
    story.append(para("核心代码包括 src/models/ 下的模型定义，src/datasets/ 下的数据读取，src/utils/ 下的指标、可视化和空间占用分析，以及 scripts/ 下的训练、评估和自采集演示入口。", styles))
    story.append(heading("附录B 项目运行命令", styles, 1))
    commands = [
        "python -m compileall src scripts tests",
        "pytest -q",
        "训练入口：python scripts/train.py --data_dir data/nyu5 --variant rgbd_concat ...",
        "自采集演示入口：python scripts/predict_campus_demo.py --rgb_dir data/campus_demo/rgb ...",
    ]
    for command in commands:
        story.append(Paragraph(command, styles["body_no_indent"]))


def make_table(caption: str, headers: list[str], rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> KeepTogether:
    table_data = [[Paragraph(cell, styles["body_no_indent"]) for cell in headers]]
    for row in rows:
        table_data.append([Paragraph(cell, styles["body_no_indent"]) for cell in row])
    table = Table(table_data, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SimSun"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return KeepTogether([Paragraph(caption, styles["table_caption"]), table, Spacer(1, 0.2 * cm)])


def scaled_image(path: str, max_width: float, max_height: float) -> Image:
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    with PILImage.open(full_path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(full_path), width * scale, height * scale)


def add_portrait_figure(story: list, path: str, caption: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(PageBreak())
    max_width = A4[0] - 4 * cm
    max_height = A4[1] - 5.0 * cm
    story.append(KeepTogether([scaled_image(path, max_width, max_height), Paragraph(caption, styles["caption"])]))


def add_inline_figure(
    story: list,
    path: str,
    caption: str,
    styles: dict[str, ParagraphStyle],
    max_height: float,
) -> None:
    max_width = A4[0] - 4 * cm
    story.append(Spacer(1, 0.25 * cm))
    story.append(KeepTogether([scaled_image(path, max_width, max_height), Paragraph(caption, styles["caption"])]))
    story.append(Spacer(1, 0.2 * cm))


def add_two_figures_page(
    story: list,
    figures: list[tuple[str, str]],
    styles: dict[str, ParagraphStyle],
) -> None:
    story.append(PageBreak())
    max_width = A4[0] - 4 * cm
    max_height = (A4[1] - 7.0 * cm) / 2
    group = []
    for path, caption in figures:
        group.append(scaled_image(path, max_width, max_height))
        group.append(Paragraph(caption, styles["caption"]))
        group.append(Spacer(1, 0.15 * cm))
    story.append(KeepTogether(group))


def add_landscape_figure(story: list, path: str, caption: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(NextPageTemplate("landscape"))
    story.append(PageBreak())
    max_width = landscape(A4)[0] - 4 * cm
    max_height = landscape(A4)[1] - 5.0 * cm
    story.append(KeepTogether([scaled_image(path, max_width, max_height), Paragraph(caption, styles["caption"])]))
    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def write_latex(assets: dict[str, str]) -> None:
    campus_keys = sorted(key for key in assets if key.startswith("campus_"))
    campus_figures = "\n".join(
        [
            rf"""\begin{{figure}}[H]\centering
\includegraphics[width=0.92\linewidth]{{../{assets[key]}}}
\caption{{自采集校园场景 {int(key.removeprefix("campus_")):03d}}}
\end{{figure}}"""
            for key in campus_keys
        ]
    )
    tex = rf"""
\documentclass[UTF8,a4paper,12pt]{{ctexart}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{setspace}}
\usepackage{{pdflscape}}
\usepackage{{float}}
\usepackage{{hyperref}}
\setstretch{{1.5}}
\hypersetup{{hidelinks}}
\title{{面向校园室内空间巡检的轻量 RGB-D 语义分割与空间占用分析系统}}
\author{{易唯 \\ 学号：202231116020106 \\ 计算机科学与技术2022级}}
\date{{2026年6月}}
\begin{{document}}
\maketitle
\thispagestyle{{empty}}
\newpage
\section*{{摘 要}}
本课程设计关注校园室内空间巡检中的通道遮挡和物品堆放问题。项目使用 NYUDepthV2 数据集，将原始室内标签整理为 other、floor、wall、obstacle、door\_window 五类，并实现轻量 RGB-D 语义分割模型 CampusDepthSegLite。实验比较 RGB-only、RGBD-concat、RGBD-boundary 和 RGBD-concat-boundary 四种变体。结果显示，RGBD-concat 整体最好，test mIoU 为 0.4683；RGBD-concat-boundary 的 obstacle IoU 最高，为 0.4899。自采集校园图像仅用于定性展示，不参与训练和定量评价。

\textbf{{关键词：}} RGB-D 语义分割；校园巡检；空间占用分析；轻量模型；卷积神经网络
\newpage
\tableofcontents
\newpage

\section{{引言}}
这次课程设计的出发点比较直观：校园室内空间中经常会出现桌椅、箱包或杂物临时占用通道的情况。人工巡检可以发现这些问题，但记录方式不够稳定，也很难量化地面可见率和障碍物比例。结合课程中关于人工智能应用边界的讨论，本项目没有把模型作为安全决策系统，而是把它定位为巡检辅助工具。

\section{{相关理论与课程知识对应}}
课程中的机器学习基础部分强调数据、模型、损失函数、优化器和评价指标构成一条完整流程。本项目也沿着这条流程完成：从 NYUDepthV2 导出 RGB-D 样本，设计轻量分割模型，使用 CrossEntropyLoss 与 DiceLoss 训练，并用 mIoU、Pixel Acc、Mean Acc 和 per-class IoU 评价。卷积神经网络课件中介绍的局部感受野和权重共享思想，是本项目选择轻量 CNN encoder 的主要课程基础。

\section{{数据集与预处理}}
本项目从 nyu\_depth\_v2\_labeled.mat 导出 1449 张样本，其中 train 1014 张、val 217 张、test 218 张。输入尺寸统一为 240 x 320。五类标签分别为 other、floor、wall、obstacle 和 door\_window。

\begin{{figure}}[H]\centering
\includegraphics[width=0.9\linewidth]{{../{assets["class_dist"]}}}
\caption{{NYU5 五类标签像素分布}}
\end{{figure}}

\begin{{figure}}[H]\centering
\includegraphics[height=0.78\textheight]{{../{assets["train_gallery"]}}}
\caption{{训练集样例可视化（报告放大版）}}
\end{{figure}}

\section{{模型设计与系统实现}}
CampusDepthSegLite 由四阶段轻量编码器和 Weighted-FPN 解码器构成。编码器通道为 [48, 96, 192, 384]，解码器通道为 128。四种模型变体分别用于验证 RGB、完整 depth、深度边界和二者组合的作用。

\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{../{assets["architecture"]}}}
\caption{{CampusDepthSegLite 系统结构示意图}}
\end{{figure}}

\begin{{table}}[H]\centering
\caption{{模型复杂度统计}}
\begin{{tabular}}{{lcc}}\toprule
Variant & Params & Params (M) \\\midrule
rgb & 1,511,689 & 1.5117 M \\
rgbd\_concat & 1,512,121 & 1.5121 M \\
rgbd\_boundary & 1,512,413 & 1.5124 M \\
rgbd\_concat\_boundary & 1,512,845 & 1.5128 M \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{实验结果与分析}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.82\linewidth]{{../{assets["curves"]}}}
\caption{{RGBD-concat 训练曲线汇总}}
\end{{figure}}

\begin{{table}}[H]\centering
\caption{{四种模型变体测试集结果}}
\begin{{tabular}}{{llllccc}}\toprule
Method & Input & Fusion & mIoU & Pixel Acc & Mean Acc & Test Loss \\\midrule
RGB-only & RGB & 无 & 0.4226 & 0.5964 & 0.5881 & 1.5588 \\
RGBD-concat & RGB+Depth & 直接拼接 & 0.4683 & 0.6318 & 0.6294 & 1.4205 \\
RGBD-boundary & RGB+Depth & 深度边界残差 & 0.4206 & 0.5974 & 0.5925 & 1.5317 \\
RGBD-concat-boundary & RGB+Depth & 拼接+边界残差 & 0.4605 & 0.6271 & 0.6191 & 1.4378 \\
\bottomrule
\end{{tabular}}
\end{{table}}

RGBD-concat 是整体最佳模型，mIoU 为 0.4683，相比 RGB-only 提升 4.57 个百分点。RGBD-concat-boundary 的 obstacle IoU 最高，说明深度边界对障碍区域有帮助，但它整体没有超过 RGBD-concat。

\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{../{assets["method_top"]}}}
\caption{{四种方法预测对比（一，报告放大版）}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{../{assets["method_bottom"]}}}
\caption{{四种方法预测对比（二，报告放大版）}}
\end{{figure}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.78\linewidth]{{../{assets["confusion"]}}}
\caption{{最佳模型归一化混淆矩阵}}
\end{{figure}}

\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{../{assets["best_cases"]}}}
\caption{{测试集中表现较好的样例}}
\end{{figure}}

\section{{自采集校园场景定性展示}}
自采集图片没有像素级 GT，也没有真实 depth，因此不参与训练、验证或测试，不计算 mIoU、Pixel Acc 或 Mean Acc。这里使用 RGB-only checkpoint，只展示系统可视化流程。

{campus_figures}

\section{{总结与展望}}
本课程设计完成了数据准备、模型设计、消融实验、空间占用分析和真实校园照片定性展示。当前结果说明完整 depth 信息比单纯深度边界更加稳定，边界信息对 obstacle 有帮助但不是整体最优。后续可以采集带真实 depth 和像素级标注的校园数据，并把风险评分规则设计得更细。

\begin{{thebibliography}}{{99}}
\bibitem{{nyu}} Silberman N, Hoiem D, Kohli P, et al. Indoor segmentation and support inference from RGBD images[C]//ECCV. 2012.
\bibitem{{fcn}} Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//CVPR. 2015.
\bibitem{{unet}} Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation[C]//MICCAI. 2015.
\bibitem{{deeplab}} Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//ECCV. 2018.
\bibitem{{fpn}} Lin T Y, Dollar P, Girshick R, et al. Feature pyramid networks for object detection[C]//CVPR. 2017.
\bibitem{{resnet}} He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//CVPR. 2016.
\bibitem{{segnet}} Badrinarayanan V, Kendall A, Cipolla R. SegNet[J]. IEEE TPAMI, 2017.
\bibitem{{fusenet}} Hazirbas C, Ma L, Domokos C, et al. FuseNet[C]//ACCV. 2016.
\bibitem{{esanet}} Seichter D, Kohler M, Lewandowski B, et al. Efficient RGB-D semantic segmentation for indoor scene analysis[C]//ICRA. 2021.
\bibitem{{pytorch}} Paszke A, Gross S, Massa F, et al. PyTorch[C]//NeurIPS. 2019.
\bibitem{{adam}} Kingma D P, Ba J. Adam: a method for stochastic optimization[C]//ICLR. 2015.
\bibitem{{dice}} Milletari F, Navab N, Ahmadi S A. V-Net[C]//3DV. 2016.
\bibitem{{course1}} 王雷春. 第1章 人工智能导论[Z]. 《人工智能技术与应用》课程课件, 2026.
\bibitem{{course2}} 王雷春. 第2章 机器学习基础[Z]. 《人工智能技术与应用》课程课件, 2026.
\bibitem{{course3}} 王雷春. 第3章 神经网络基础[Z]. 《人工智能技术与应用》课程课件, 2026.
\bibitem{{course4}} 王雷春. 第4章 卷积神经网络及应用[Z]. 《人工智能技术与应用》课程课件, 2026.
\end{{thebibliography}}
\end{{document}}
"""
    LATEX_PATH.write_text(tex.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
