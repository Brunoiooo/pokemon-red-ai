#!/usr/bin/env python3
"""Generuje prezentacja.pptx (obrona pracy inżynierskiej) z obrazów w img/
i notatek prelegenta w notatki.json.

Wymaga: python-pptx, Pillow.
Użycie: python docs/prezentacja/build_pptx.py
Geometria jest zapisana w pikselach płótna 1920×1080 (16:9) i przeliczana
na cale (144 px = 1 cal), a rozmiary czcionek w pt = px / 2.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"
OUT = HERE / "prezentacja.pptx"
NOTES = json.loads((HERE / "notatki.json").read_text(encoding="utf-8"))

# --- paleta -----------------------------------------------------------------
INK = "13213B"
PAPER = "F6F5F1"
PAPER2 = "ECEAE3"
CARD = "FBFBF8"
LINE = "DCD8CD"
ACCENT = "2A78D6"
ACCENT2 = "B85E12"
MUTED = "4A5568"
FOOT = "8A8F99"
ONDARK = "C9D3E3"
DARK_EYEBROW = "7FB2F0"
PILL_BG = "DDE9F8"
PILL_INK = "1B4F8F"

SANS = "Calibri"
MONO = "Consolas"

EMU_PER_PX = 914400 / 144  # 1920 px = 13,333 cala


def px(v: float) -> Emu:
    return Emu(int(round(v * EMU_PER_PX)))


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width = px(1920)
prs.slide_height = px(1080)
BLANK = prs.slide_layouts[6]
FOOTER_LABEL = "Agent PPO · Pokémon Red"


# --- prymitywy ---------------------------------------------------------------
def background(slide, color: str) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)


def text(slide, x, y, w, h, paras, *, anchor=MSO_ANCHOR.TOP, margin=0, name=None):
    """paras: lista akapitów; akapit = dict(runs=[(tekst, opcje)], align, space_after, line)
    albo skrót (tekst, opcje)."""
    tb = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    if name:
        tb.name = name
    fill_frame(tb.text_frame, paras, anchor=anchor, margin=margin)
    return tb


def fill_frame(tf, paras, *, anchor=MSO_ANCHOR.TOP, margin=0):
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    m = px(margin) if not isinstance(margin, tuple) else None
    if isinstance(margin, tuple):
        tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (px(v) for v in margin)
    else:
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = m
    if isinstance(paras, (tuple, dict)):
        paras = [paras]
    first = True
    for para in paras:
        if isinstance(para, tuple):
            para = {"runs": [para]}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = para.get("align", PP_ALIGN.LEFT)
        if "space_after" in para:
            p.space_after = Pt(para["space_after"])
        if "space_before" in para:
            p.space_before = Pt(para["space_before"])
        if "line" in para:
            p.line_spacing = para["line"]
        for t, o in para["runs"]:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", SANS)
            f.size = Pt(o.get("size", 16))
            f.bold = o.get("bold", False)
            f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", INK))
            if "spacing" in o:  # rozstrzelenie liter w setnych pt
                r._r.get_or_add_rPr().set("spc", str(int(o["spacing"] * 100)))


def rect(slide, x, y, w, h, *, fill=CARD, line=LINE, radius=16, line_w=1.25, dash=False, name=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(kind, px(x), px(y), px(w), px(h))
    if name:
        s.name = name
    if radius:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
        if dash:
            s.line.dash_style = MSO_LINE.DASH
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def image(slide, name, x, y, w, h, *, alt="", border=False, bg=None):
    """Wstawia obraz dopasowany (contain) do pola x,y,w,h."""
    path = IMG / name
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    dx, dy = x + (w - dw) / 2, y + (h - dh) / 2
    if border or bg:
        rect(slide, x, y, w, h, fill=bg or CARD, line=LINE if border else None, radius=12)
    pic = slide.shapes.add_picture(str(path), px(dx), px(dy), px(dw), px(dh))
    pic.name = name
    if alt:
        pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def arrow(slide, x1, y1, x2, y2, color=ACCENT, width=2.25):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(x1), px(y1), px(x2), px(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ln.append(parse_xml(
        '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'type="triangle" w="med" len="med"/>'))
    return c


def card(slide, x, y, w, h, *, tag=None, title=None, body=None, hl=False, fill=CARD,
         body_size=14, title_size=19, pad=(30, 26, 30, 22)):
    s = rect(slide, x, y, w, h, fill=fill, line=ACCENT if hl else LINE, line_w=2 if hl else 1.25)
    paras = []
    if tag:
        paras.append({"runs": [(tag, dict(font=MONO, size=12, bold=True, color=ACCENT))], "space_after": 4})
    if title:
        paras.append({"runs": [(title, dict(size=title_size, bold=True))], "space_after": 6})
    if body:
        paras.append({"runs": [(body, dict(size=body_size, color=MUTED))], "line": 1.08})
    fill_frame(s.text_frame, paras, margin=pad)
    return s


def header(slide, eyebrow, title, *, dark=False):
    text(slide, 128, 104, 1664, 30,
         (eyebrow.upper(), dict(font=MONO, size=12, bold=True, color=DARK_EYEBROW if dark else ACCENT, spacing=1.5)))
    text(slide, 128, 140, 1664, 90, (title, dict(size=34, bold=True, color=PAPER if dark else INK)))


def footer(slide, n, label=FOOTER_LABEL, dark=False):
    col = "8C9AB5" if dark else FOOT
    text(slide, 128, 1004, 1200, 30, (label, dict(font=MONO, size=12, color=col)))
    text(slide, 1592, 1004, 200, 30, {"runs": [(str(n), dict(font=MONO, size=12, color=col))], "align": PP_ALIGN.RIGHT})


def new_slide(sid, bg=PAPER):
    s = prs.slides.add_slide(BLANK)
    background(s, bg)
    s.notes_slide.notes_text_frame.text = NOTES.get(sid, "")
    return s


def bullets(slide, x, y, w, h, items, *, size=16, numbered=True, color=INK, space=8):
    tb = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for v in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, v, 0)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(px(44))))
        pPr.set("indent", str(int(-px(44))))
        bullet = ('<a:buFont xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" typeface="+mj-lt"/>'
                  '<a:buAutoNum xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="arabicPeriod"/>'
                  if numbered else
                  '<a:buChar xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" char="–"/>')
        for frag in parse_xml(f'<x xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">{bullet}</x>'):
            pPr.append(frag)
        r = p.add_run()
        r.text = it
        r.font.name = SANS
        r.font.size = Pt(size)
        r.font.color.rgb = rgb(color)
    return tb


def table(slide, x, y, w, rows, col_pct, *, size=14, row_fills=None, row_h=64):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), px(x), px(y), px(w), px(row_h * len(rows)))
    tbl = shape.table
    tblPr = tbl._tbl.tblPr
    for attr in ("bandRow", "firstRow"):
        tblPr.set(attr, "0")
    style = tblPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId")
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
    for i, pct in enumerate(col_pct):
        tbl.columns[i].width = px(w * pct)
    for r, row in enumerate(rows):
        tbl.rows[r].height = px(row_h)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = px(18)
            cell.margin_top = cell.margin_bottom = px(10)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = (row_fills or {}).get(r, CARD if r else PAPER2)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(fill)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = val
            run.font.name = SANS
            run.font.size = Pt(size)
            run.font.bold = r == 0
            run.font.color.rgb = rgb(INK)
    return shape


# =============================================================================
# Slajdy
# =============================================================================
n = 0

# 1. Okładka -------------------------------------------------------------------
s = new_slide("cover")
n += 1
image(s, "logo_pk.png", 128, 96, 124, 126, alt="Logo Politechniki Koszalińskiej")
text(s, 284, 112, 1200, 40, ("POLITECHNIKA KOSZALIŃSKA", dict(size=15, bold=True, spacing=0.5)))
text(s, 284, 156, 1200, 40, ("Wydział Elektroniki i Informatyki · kierunek Informatyka", dict(size=13, color=MUTED)))
text(s, 128, 336, 1664, 30, ("PRACA DYPLOMOWA INŻYNIERSKA", dict(font=MONO, size=12, bold=True, color=ACCENT, spacing=1.5)))
text(s, 128, 378, 1600, 320, {"runs": [("System uczenia ze wzmocnieniem do autonomicznej gry w Pokémon Red "
                                         "z wykorzystaniem algorytmu PPO", dict(size=42, bold=True))], "line": 0.95})
text(s, 128, 712, 1600, 44, ("Agent sterujący emulatorem Game Boya na podstawie pamięci RAM i obrazu ekranu",
                              dict(size=16, color=MUTED)))
rect(s, 128, 826, 1664, 2, fill=LINE, line=None, radius=0)
text(s, 128, 852, 640, 90, [("Autor", dict(size=12, color=MUTED)),
                            ("[Imię i nazwisko] · nr albumu [______]", dict(size=15, bold=True))])
text(s, 820, 852, 640, 90, [("Promotor", dict(size=12, color=MUTED)),
                            ("dr hab. inż. Adam Słowik, prof. PK", dict(size=15, bold=True))])
text(s, 1440, 880, 352, 40, {"runs": [("Koszalin, [data obrony]", dict(size=14, color=MUTED))], "align": PP_ALIGN.RIGHT})

# 2. Plan -----------------------------------------------------------------------
s = new_slide("plan", PAPER2)
n += 1
header(s, "Plan prezentacji", "Od problemu do wyników")
items = [("01", "Problem i cel", "Dlaczego gra Pokémon Red jest trudnym środowiskiem dla agenta oraz cel i zakres pracy"),
         ("02", "Rozwiązanie", "Architektura systemu, obserwacje, sieć neuronowa, funkcja nagrody i curriculum learning"),
         ("03", "Wyniki", "Przebieg uczenia, osiągnięte zdarzenia fabularne i nagranie z ewaluacji agenta"),
         ("04", "Wnioski", "Napotkane problemy, wnioski z realizacji pracy i kierunki dalszego rozwoju")]
cw = (1664 - 3 * 32) / 4
for i, (t, h, b) in enumerate(items):
    card(s, 128 + i * (cw + 32), 290, cw, 420, tag=t, title=h, body=b, fill=PAPER, body_size=15)
footer(s, n)

# 3. Problem --------------------------------------------------------------------
s = new_slide("problem")
n += 1
header(s, "Wprowadzenie", "Pokémon Red jako środowisko uczenia")
for i, (h, b) in enumerate([
        ("Rzadka nagroda", "Postęp fabularny pojawia się co tysiące naciśnięć przycisków, a gra nie podaje wyniku punktowego"),
        ("Długi horyzont", "Zdobycie pierwszej odznaki wymaga kilkunastu zdarzeń fabularnych w ściśle określonej kolejności"),
        ("Cztery tryby gry", "Świat, dialog, menu i walka — ten sam przycisk ma w każdym trybie inne znaczenie")]):
    card(s, 128, 270 + i * 212, 940, 192, title=h, body=b, body_size=15)
for i, (f, alt) in enumerate([("game_start.png", "Pokój bohatera na początku gry"),
                              ("game_starter.png", "Laboratorium profesora po wyborze pierwszego Pokémona"),
                              ("game_battle.png", "Ekran walki z dzikim Pokémonem"),
                              ("game_forest.png", "Viridian Forest — najdalszy obszar osiągnięty przez agenta")]):
    image(s, f, 1150 + (i % 2) * 330, 270 + (i // 2) * 296, 310, 279, alt=alt)
footer(s, n)

# 4. Cel i zakres ---------------------------------------------------------------
s = new_slide("cel", PAPER2)
n += 1
header(s, "Cel i zakres pracy", "Agent, który samodzielnie robi postęp w grze")
c = rect(s, 128, 280, 620, 600, fill=PAPER, line=ACCENT, line_w=2)
fill_frame(c.text_frame, [
    {"runs": [("CEL PRACY", dict(font=MONO, size=12, bold=True, color=ACCENT, spacing=1))], "space_after": 10},
    {"runs": [("Opracowanie i implementacja systemu uczenia ze wzmocnieniem, w którym agent PPO uczy się "
               "grać w Pokémon Red bez wiedzy eksperckiej o kolejności zadań", dict(size=20, bold=False))],
     "line": 1.1}], anchor=MSO_ANCHOR.MIDDLE, margin=(40, 36, 40, 36))
text(s, 820, 300, 972, 30, ("ZAKRES PRACY", dict(font=MONO, size=12, bold=True, color=ACCENT, spacing=1)))
bullets(s, 820, 350, 972, 520, [
    "analiza mapy pamięci RAM i dezasemblacji gry (pret/pokered)",
    "środowisko Gymnasium oparte na emulatorze PyBoy",
    "przestrzeń obserwacji, sieć neuronowa i maskowanie akcji",
    "wielopoziomowa funkcja nagrody z karami za pętle",
    "automatyczne curriculum oparte na zdarzeniach fabularnych",
    "panel sterowania, narzędzia diagnostyczne i ewaluacja"], size=17, space=12)
footer(s, n)

# 5. Narzędzia ------------------------------------------------------------------
s = new_slide("narzedzia")
n += 1
header(s, "Narzędzia", "Wykorzystane technologie")
tools = [("EMULACJA", "PyBoy", "Emulator Game Boya w Pythonie: bezpośredni odczyt RAM, zapisy stanu, praca bez okna"),
         ("ŚRODOWISKO", "Gymnasium", "Standardowy interfejs reset / step, przestrzenie obserwacji typu Dict i akcji Discrete"),
         ("ALGORYTM", "Stable-Baselines3", "MaskablePPO z sb3-contrib, 8 równoległych środowisk w SubprocVecEnv"),
         ("SIECI NEURONOWE", "PyTorch + CUDA", "Własny ekstraktor cech łączący dwie sieci CNN i sieć MLP"),
         ("ANALIZA", "TensorBoard, Optuna", "Ponad 70 metryk uczenia oraz automatyczne strojenie hiperparametrów"),
         ("WIEDZA O GRZE", "pret/pokered", "Dezasemblacja gry: mapa RAM, skrypty map, flagi zdarzeń, warpy i kolizje")]
cw = (1664 - 2 * 32) / 3
for i, (t, h, b) in enumerate(tools):
    card(s, 128 + (i % 3) * (cw + 32), 280 + (i // 3) * 330, cw, 300, tag=t, title=h, body=b, body_size=15)
footer(s, n)

# 6. Architektura ---------------------------------------------------------------
s = new_slide("architektura", PAPER2)
n += 1
header(s, "Architektura", "Pętla agent — środowisko")
ox, oy = 128, 280
boxes = [(0, 0, 340, "ROM + PYBOY", "Emulator", "Naciśnięcie przycisku, 16 klatek na akcję", False),
         (440, 0, 360, "DATA.PY", "Odczyt RAM", "Tryb gry, drużyna, walka, mapa, flagi zdarzeń", False),
         (900, 0, 360, "POKEMONREDENV", "Środowisko", "Obserwacja, nagroda, maska akcji, koniec epizodu", True),
         (1360, 0, 304, "MASKABLEPPO", "Agent", "8 procesów, wspólna polityka", False)]
for bx, by, bw, t, h, b, hl in boxes:
    card(s, ox + bx, oy + by, bw, 262, tag=t, title=h, body=b, hl=hl, fill=PAPER, body_size=14)
card(s, ox + 440, oy + 360, 820, 232, tag="CURRICULUM + MILESTONECALLBACK", title="Zapisy stanu i wybór celu",
     body="Każde nowe zdarzenie fabularne zapisuje punkt startowy; kolejny cel jest dobierany na podstawie skuteczności",
     fill=PAPER, body_size=15)
card(s, ox + 1360, oy + 360, 304, 232, tag="LOGI", title="TensorBoard", body="Metryki i modele", fill=PAPER, body_size=15)
arrow(s, ox + 340, oy + 120, ox + 440, oy + 120)
arrow(s, ox + 800, oy + 120, ox + 900, oy + 120)
arrow(s, ox + 1260, oy + 90, ox + 1360, oy + 90)
arrow(s, ox + 1360, oy + 160, ox + 1260, oy + 160, color=ACCENT2)
arrow(s, ox + 1080, oy + 262, ox + 1080, oy + 360)
arrow(s, ox + 1512, oy + 262, ox + 1512, oy + 360)
text(s, ox + 1272, oy + 44, 90, 30, ("obs.", dict(size=12, color=MUTED)))
text(s, ox + 1268, oy + 168, 90, 30, ("akcja", dict(size=12, color=ACCENT2)))
footer(s, n)

# 7. Obserwacje -----------------------------------------------------------------
s = new_slide("obserwacje")
n += 1
header(s, "Przestrzeń obserwacji", "Co widzi agent")
image(s, "obs_world.png", 128, 262, 1664, 450, border=True,
      alt="Ekran emulatora w Viridian Forest obok siatki screen_tiles 18×20 i mapy odwiedzin visit_mask 11×11")
obs = [("screen_tiles · 1×18×20", "Identyfikatory kafelków ekranu odczytane z RAM zamiast pikseli — 72 razy mniej danych niż obraz"),
       ("visit_mask · 1×11×11", "Lokalna mapa liczby odwiedzin pól wokół gracza — informuje, gdzie agent już był"),
       ("vector · 2059 cech", "Drużyna, walka, menu, ekwipunek, histogramy map i dialogów, kompas do celu, cel w kodzie one-hot")]
cw = (1664 - 2 * 40) / 3
for i, (pill, b) in enumerate(obs):
    x = 128 + i * (cw + 40)
    p = rect(s, x, 740, 330, 44, fill=PILL_BG, line=None, radius=22)
    fill_frame(p.text_frame, {"runs": [(pill, dict(font=MONO, size=12, bold=True, color=PILL_INK))],
                              "align": PP_ALIGN.CENTER}, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, 800, cw, 150, {"runs": [(b, dict(size=14, color=MUTED))], "line": 1.1})
footer(s, n)

# 8. Sieć ----------------------------------------------------------------------
s = new_slide("siec", PAPER2)
n += 1
header(s, "Model", "Sieć neuronowa i przestrzeń akcji")
ox, oy = 128, 280
for i, lab in enumerate(["screen_tiles", "visit_mask", "vector (2059)"]):
    p = rect(s, ox, oy + 30 + i * 200, 280, 48, fill=PILL_BG, line=None, radius=24)
    fill_frame(p.text_frame, {"runs": [(lab, dict(font=MONO, size=12, bold=True, color=PILL_INK))],
                              "align": PP_ALIGN.CENTER}, anchor=MSO_ANCHOR.MIDDLE)
branches = [("CNN", "3 warstwy splotowe: 32, 64, 64 filtry"),
            ("CNN", "2 warstwy splotowe: 16, 32 filtry"),
            ("MLP", "2 warstwy gęste: 256, 128 neuronów")]
for i, (h, b) in enumerate(branches):
    card(s, ox + 360, oy + i * 200, 400, 164, title=h, body=b, fill=PAPER, title_size=17, body_size=13, pad=(26, 18, 26, 14))
card(s, ox + 860, oy + 160, 330, 176, title="Fuzja", body="Konkatenacja i warstwa liniowa 256 + ReLU", hl=True,
     fill=PAPER, title_size=17, body_size=13, pad=(26, 18, 26, 14))
card(s, ox + 1290, oy + 0, 374, 196, title="Polityka π", body="MLP 256-256 → 9 akcji, maska niedozwolonych przycisków",
     fill=PAPER, title_size=17, body_size=13, pad=(26, 18, 26, 14))
card(s, ox + 1290, oy + 300, 374, 164, title="Wartość V", body="MLP 256-256 → ocena stanu", fill=PAPER,
     title_size=17, body_size=13, pad=(26, 18, 26, 14))
for i in range(3):
    arrow(s, ox + 280, oy + 54 + i * 200, ox + 360, oy + 70 + i * 200)
arrow(s, ox + 760, oy + 80, ox + 860, oy + 210)
arrow(s, ox + 760, oy + 280, ox + 860, oy + 250)
arrow(s, ox + 760, oy + 480, ox + 860, oy + 290)
arrow(s, ox + 1190, oy + 210, ox + 1290, oy + 100)
arrow(s, ox + 1190, oy + 290, ox + 1290, oy + 380)
text(s, ox + 860, oy + 510, 804, 90, {"runs": [("Akcje: A, B, Start, Select, 4 kierunki, brak akcji · 16 klatek na akcję · "
                                               "γ = 0,99 · lr = 2,5·10⁻⁴", dict(size=13, color=MUTED))], "line": 1.1})
footer(s, n)

# 9. Nagroda -------------------------------------------------------------------
s = new_slide("nagroda")
n += 1
header(s, "Funkcja nagrody", "Trzy poziomy nagrody")
table(s, 128, 270, 1664, [
    ["Poziom", "Składniki", "Charakter"],
    ["Makro", "nowa flaga zdarzenia fabularnego lub odznaka, leczenie w nowym mieście, kara za regres", "jednorazowa, duża"],
    ["Mezo", "wynik walki (wygrana, ucieczka, porażka), nowa mapa, nowy dialog, poziomy drużyny", "malejąca z powtórzeniami"],
    ["Mikro", "koszt kroku, kary za pętle ruchu, bezczynność w menu, spam przycisków, ponowne otwieranie dialogów",
     "ciągła, mała, ujemna"]], [0.16, 0.56, 0.28], size=15, row_h=96,
    row_fills={0: PAPER2, 1: "E3EDF9", 2: CARD, 3: "F3E6D9"})
c = rect(s, 128, 720, 1664, 130)
fill_frame(c.text_frame, {"runs": [("Każdy składnik jest logowany osobno (17 składowych w TensorBoard), co pozwoliło wykryć "
                                    "i naprawić nadużycia nagrody, np. farmienie walk z dzikimi Pokémonami",
                                    dict(size=15, color=MUTED))], "line": 1.1},
           anchor=MSO_ANCHOR.MIDDLE, margin=(40, 20, 40, 20))
footer(s, n)

# 10. Curriculum ---------------------------------------------------------------
s = new_slide("curriculum", PAPER2)
n += 1
header(s, "Curriculum learning", "Automatyczny dobór celów treningowych")
steps = ["471 celów to flagi zdarzeń z pamięci gry i 8 odznak; graf zależności wyznaczono analizą statyczną skryptów pret/pokered",
         "Osiągnięcie nowego zdarzenia zapisuje stan emulatora, który staje się nowym punktem startowym epizodów",
         "Gdy 70% ostatnich epizodów osiąga cel, przydzielany jest nowy — najtrudniejszy spośród jeszcze nieopanowanych"]
for i, b in enumerate(steps):
    y = 270 + i * 200
    rect(s, 128, y, 820, 176, fill=PAPER)
    text(s, 156, y + 24, 60, 100, ("%d" % (i + 1), dict(size=36, bold=True, color=ACCENT)))
    text(s, 228, y + 30, 690, 130, {"runs": [(b, dict(size=15, color=MUTED))], "line": 1.1})
image(s, "compass.png", 996, 270, 796, 448, border=True,
      alt="Narzędzie Event Compass: punkty wyzwalania zdarzeń naniesione na mapę lokacji")
text(s, 996, 734, 796, 90, {"runs": [("Event Compass — miejsca wyzwalania zdarzeń wyznaczone z danych map; ten sam kompas "
                                      "trafia do wektora obserwacji", dict(size=13, color=MUTED))], "line": 1.1})
footer(s, n)

# 11. Analiza ------------------------------------------------------------------
s = new_slide("analiza")
n += 1
header(s, "Narzędzia diagnostyczne", "Wiedza o grze wygenerowana z dezasemblacji")
tools = [("map_pins.png", "Map Pins", "1128 punktów wyzwalania zdarzeń naniesionych na grafikę 98 map"),
         ("script_graph.png", "Script Graph", "Automaty stanów skryptów map (202 stany) z warunkami i ustawianymi flagami"),
         ("route_map.png", "Kanto Pathfinder", "Graf świata: 220 map i 795 przejść, najkrótsza trasa do celu")]
cw = (1664 - 2 * 40) / 3
for i, (f, h, b) in enumerate(tools):
    x = 128 + i * (cw + 40)
    image(s, f, x, 280, cw, cw * 9 / 16, border=True, alt=h)
    text(s, x, 280 + cw * 9 / 16 + 24, cw, 54, (h, dict(size=19, bold=True)))
    text(s, x, 280 + cw * 9 / 16 + 82, cw, 110, {"runs": [(b, dict(size=14, color=MUTED))], "line": 1.1})
footer(s, n)

# 12. Panel sterowania ---------------------------------------------------------
s = new_slide("gui", PAPER2)
n += 1
header(s, "Aplikacja", "Panel sterowania")
image(s, "gui.png", 128, 270, 960, 600, border=True,
      alt="Panel sterowania w przeglądarce: formularz parametrów uczenia i podgląd polecenia")
for i, (h, b) in enumerate([("Pięć trybów", "train, eval, benchmark-hparams, save-stage, debug-play"),
                            ("Formularz z argparse", "Każdy parametr CLI pojawia się automatycznie z wartością domyślną i opisem"),
                            ("Podgląd na żywo", "Gotowe polecenie i strumień wyjścia procesu w przeglądarce")]):
    card(s, 1136, 270 + i * 210, 656, 192, title=h, body=b, fill=PAPER, body_size=15)
footer(s, n)

# 13. Przebieg uczenia ---------------------------------------------------------
s = new_slide("uczenie")
n += 1
header(s, "Wyniki", "Przebieg uczenia")
image(s, "chart_training.png", 128, 270, 1104, 600, border=True,
      alt="Cztery wykresy w funkcji kroków uczenia: średnia nagroda, odsetek epizodów z osiągniętym celem, "
          "liczba zdarzeń fabularnych i liczba odwiedzonych map w epizodzie")
for i, (big, lab) in enumerate([("9,66 mln", "kroków uczenia w ~6,8 h (8 środowisk, ~400 kroków/s)"),
                                ("0 → 15", "zdarzeń fabularnych w jednym epizodzie"),
                                ("93%", "epizodów kończących się osiągnięciem celu (koniec uczenia)")]):
    y = 280 + i * 200
    text(s, 1280, y, 512, 100, (big, dict(size=38, bold=True, color=ACCENT)))
    text(s, 1280, y + 100, 512, 90, {"runs": [(lab, dict(size=14, color=MUTED))], "line": 1.1})
footer(s, n, FOOTER_LABEL + " · przebieg ppo_20260915")

# 14. Osiągnięte cele ----------------------------------------------------------
s = new_slide("cele", PAPER2)
n += 1
header(s, "Wyniki", "Osiągnięte zdarzenia fabularne")
image(s, "chart_goals.png", 128, 270, 1104, 600, border=True,
      alt="Wykres słupkowy liczby osiągnięć 17 kolejnych zdarzeń fabularnych, od około 2000 dla pierwszych "
          "zdarzeń do 1 dla trzeciego trenera w Viridian Forest")
for i, (big, lab) in enumerate([("20", "zdarzeń z automatycznie zapisanym stanem gry"),
                                ("16", "map odwiedzanych w jednym epizodzie — od domu bohatera po las")]):
    y = 270 + i * 216
    c = rect(s, 1280, y, 512, 206, fill=PAPER)
    fill_frame(c.text_frame, [{"runs": [(big, dict(size=34, bold=True, color=ACCENT))]},
                              {"runs": [(lab, dict(size=14, color=MUTED))], "line": 1.0}], margin=(30, 16, 30, 12))
card(s, 1280, 712, 512, 196, title="Najdalszy postęp",
     body="Rywal na Route 22 i trzej trenerzy w Viridian Forest; pierwszej odznaki jeszcze brak",
     fill=PAPER, body_size=13, pad=(30, 18, 30, 14))
footer(s, n)

# 15. Demonstracja -------------------------------------------------------------
s = new_slide("demo", INK)
n += 1
header(s, "Demonstracja", "Agent w działaniu — nagranie z ewaluacji", dark=True)
d = rect(s, 128, 270, 1000, 600, fill=None, line="5A6B8C", dash=True, line_w=2)
tri = s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, px(588), px(420), px(80), px(80))
tri.rotation = 90
tri.fill.solid()
tri.fill.fore_color.rgb = rgb(DARK_EYEBROW)
tri.line.fill.background()
fill_frame(d.text_frame, [{"runs": [("", dict(size=12, color=PAPER))], "space_after": 110},
                          {"runs": [("[miejsce na nagranie z ewaluacji]", dict(size=17, bold=True, color=PAPER))],
                           "align": PP_ALIGN.CENTER, "space_after": 8},
                          {"runs": [("python cli.py eval --gui", dict(font=MONO, size=12, color="8C9AB5"))],
                           "align": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE)
image(s, "obs_battle.png", 1176, 270, 616, 189, bg=CARD,
      alt="Walka z dzikim Pokémonem: ekran emulatora i obserwacja agenta w trakcie walki")
text(s, 1176, 490, 616, 100, {"runs": [("Model z 9,66 mln kroków, polityka stochastyczna z maskowaniem akcji",
                                        dict(size=15, color=ONDARK))], "line": 1.1})
text(s, 1176, 600, 616, 100, {"runs": [("Po każdym osiągniętym celu epizod trwa dalej z nowym, losowo dobranym celem",
                                        dict(size=15, color=ONDARK))], "line": 1.1})
footer(s, n, dark=True)

# 16. Problemy -----------------------------------------------------------------
s = new_slide("problemy")
n += 1
header(s, "Napotkane problemy", "Nadużycia nagrody i ich rozwiązania")
table(s, 128, 270, 1664, [
    ["Zaobserwowane zachowanie", "Zastosowane rozwiązanie"],
    ["Chodzenie w kółko, spam przycisków w menu", "kary za pętle i bezczynność, maskowanie akcji, przerywanie zablokowanych epizodów"],
    ["Farmienie walk z dzikimi Pokémonami", "nagroda malejąca z liczbą walk w danym miejscu, łagodniejsze skalowanie trudności"],
    ["Zbyt słaba drużyna przed salą lidera", "nagroda za poziomy drużyny z podniesionym progiem nasycenia"],
    ["Zdarzenia osiągane w innej kolejności niż w scenariuszu",
     "curriculum bez stałej kolejności, zapis stanu dla każdego osiągniętego zdarzenia"]],
    [0.4, 0.6], size=16, row_h=104, row_fills={0: PAPER2, 1: CARD, 2: "EFEDE7", 3: CARD, 4: "EFEDE7"})
footer(s, n)

# 17. Wnioski ------------------------------------------------------------------
s = new_slide("wnioski", PAPER2)
n += 1
header(s, "Podsumowanie i wnioski", "Cel pracy został osiągnięty")
bullets(s, 128, 280, 1560, 640, [
    "Opracowano kompletny system uczenia ze wzmocnieniem dla gry Pokémon Red: środowisko, model, curriculum i narzędzia.",
    "Agent PPO bez wiedzy o kolejności zadań samodzielnie osiągnął 20 zdarzeń fabularnych i dotarł do Viridian Forest.",
    "Odczyt stanu z pamięci RAM i maskowanie akcji pozwoliły prowadzić uczenie na pojedynczym komputerze (~400 kroków/s).",
    "Automatyczne curriculum oparte na zapisach stanu skraca horyzont zadania i ułatwia dalsze uczenie.",
    "Projekt funkcji nagrody wymaga stałej obserwacji jej składowych — agent szybko wykorzystuje każdą lukę."],
    size=19, space=18)
footer(s, n)

# 18. Dalszy rozwój ------------------------------------------------------------
s = new_slide("rozwoj")
n += 1
header(s, "Dalszy rozwój", "Kierunki dalszych prac")
dev = [("POSTĘP", "Pierwsza odznaka i dalej", "Dłuższe uczenie od zapisanych stanów w Viridian Forest, aż do walki z liderem sali w Pewter City"),
       ("MODEL", "Pamięć w polityce", "Sieć rekurencyjna (LSTM) lub transformer, aby agent pamiętał kontekst dłuższy niż pojedyncza obserwacja"),
       ("OPTYMALIZACJA", "Algorytmy ewolucyjne", "Porównanie PPO z neuroewolucją oraz ewolucyjne strojenie wag składowych nagrody"),
       ("UOGÓLNIENIE", "Inne gry Game Boya", "Przeniesienie generatorów wiedzy z dezasemblacji na Pokémon Blue, Yellow i podobne gry")]
cw = (1664 - 32) / 2
for i, (t, h, b) in enumerate(dev):
    card(s, 128 + (i % 2) * (cw + 32), 280 + (i // 2) * 320, cw, 290, tag=t, title=h, body=b, body_size=16)
footer(s, n)

# 19. Zakończenie --------------------------------------------------------------
s = new_slide("koniec", INK)
n += 1
text(s, 128, 330, 1664, 30, ("AGENT PPO · POKÉMON RED", dict(font=MONO, size=12, bold=True, color=DARK_EYEBROW, spacing=1.5)))
text(s, 128, 372, 1664, 160, ("Dziękuję za uwagę", dict(size=60, bold=True, color=PAPER)))
text(s, 128, 540, 1664, 60, ("Zapraszam do pytań", dict(size=20, color=ONDARK)))
rect(s, 128, 640, 1100, 2, fill="2E3D5C", line=None, radius=0)
text(s, 128, 668, 1400, 50, ("[Imię i nazwisko]      Promotor: dr hab. inż. Adam Słowik, prof. PK", dict(size=14, color=ONDARK)))

prs.core_properties.title = "System uczenia ze wzmocnieniem do autonomicznej gry w Pokémon Red z wykorzystaniem algorytmu PPO"
prs.core_properties.subject = "Prezentacja pracy dyplomowej inżynierskiej"
prs.save(OUT)
print(f"zapisano {OUT} ({n} slajdów)")
