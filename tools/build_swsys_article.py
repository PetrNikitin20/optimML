from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "factorial_pilot"
OUTPUT = ROOT / "article" / "SWSYS_RU_pairwise_pointwise_real_data.docx"


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def set_run_font(run, size: float = 11, bold: bool | None = None, italic: bool | None = None):
    run.font.name = "Times New Roman"
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), "Times New Roman")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_margins(cell, value: int = 70):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge in ("top", "start", "bottom", "end"):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color: str = "808080"):
    tbl_pr = table._tbl.tblPr
    old = tbl_pr.find(qn("w:tblBorders"))
    if old is not None:
        tbl_pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), color)
        borders.append(node)
    tbl_pr.append(borders)


def shade(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_body(doc: Document, text: str, *, first_line: bool = True, italic: bool = False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)
    if first_line:
        p.paragraph_format.first_line_indent = Cm(1.0)
    set_run_font(p.add_run(text), 11, italic=italic)
    return p


def add_heading(doc: Document, text: str, level: int = 1):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8 if level == 1 else 5)
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    set_run_font(p.add_run(text), 12 if level == 1 else 11, bold=True)
    return p


def add_labeled(doc: Document, label: str, text: str, *, english: bool = False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(5)
    set_run_font(p.add_run(label), 10, bold=True)
    set_run_font(p.add_run(text), 10, italic=english)
    return p


def add_equation(doc: Document, formula: str, number: int):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)
    table.columns[0].width = Cm(15.3)
    table.columns[1].width = Cm(1.2)
    for cell in table.rows[0].cells:
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            node = OxmlElement(f"w:{edge}")
            node.set(qn("w:val"), "nil")
            borders.append(node)
        tc_pr.append(borders)
    p = table.cell(0, 0).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    math_para = OxmlElement("m:oMathPara")
    math = OxmlElement("m:oMath")
    run = OxmlElement("m:r")
    text = OxmlElement("m:t")
    text.text = formula
    run.append(text)
    math.append(run)
    math_para.append(math)
    p._p.append(math_para)
    pn = table.cell(0, 1).paragraphs[0]
    pn.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_run_font(pn.add_run(f"({number})"), 11)
    return table


def add_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]], widths: list[float]):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.space_before = Pt(5)
    cap.paragraph_format.space_after = Pt(3)
    set_run_font(cap.add_run(caption), 10, bold=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)
    for idx, value in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.width = Cm(widths[idx])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        shade(cell, "D9E2F3")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        set_run_font(p.add_run(value), 8, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].width = Cm(widths[idx])
            cells[idx].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cells[idx])
            p = cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            set_run_font(p.add_run(str(value)), 8)
    note = doc.add_paragraph()
    note.paragraph_format.space_after = Pt(4)
    set_run_font(note.add_run("Примечание. Значения представлены как среднее ± выборочное стандартное отклонение по трём seed."), 8, italic=True)
    return table


def fmt(mean: float, sd: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}".replace(".", ",")


def metric(summary: dict, key: str, digits: int = 4) -> str:
    item = summary["metrics"][key]
    return fmt(item["mean"], item["sd"], digits)


def delta(summary: dict, key: str, digits: int = 4) -> str:
    item = summary["metrics"][key]
    return fmt(item["mean_delta"], item["sd_delta"], digits)


def configure_document(doc: Document):
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(11)


def add_reference(doc: Document, number: int, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.first_line_indent = Cm(-0.7)
    p.paragraph_format.space_after = Pt(2)
    set_run_font(p.add_run(f"{number}. {text}"), 9)


def build():
    pair0 = read_json("pairwise_3B_ultrafeedback_noise0.0_summary.json")
    point0 = read_json("pointwise_3B_ultrafeedback_noise0.0_summary.json")
    pair1 = read_json("pairwise_3B_ultrafeedback_noise0.1_summary.json")
    point1 = read_json("pointwise_3B_ultrafeedback_noise0.1_summary.json")
    contrast0 = read_json("paired_loss_contrast_3B_ultrafeedback_noise0.0_summary.json")
    contrast1 = read_json("paired_loss_contrast_3B_ultrafeedback_noise0.1_summary.json")

    doc = Document()
    configure_document(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_run_font(p.add_run("УДК 004.8:519.6"), 11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(8)
    title.paragraph_format.space_after = Pt(8)
    set_run_font(title.add_run("СРАВНЕНИЕ ПОПАРНОЙ И ПОЭЛЕМЕНТНОЙ ОПТИМИЗАЦИИ ПРЕДПОЧТЕНИЙ ЯЗЫКОВОЙ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ"), 14, bold=True)

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(author.add_run("Никитин Петр Владимирович, к.пед.н., доцент"), 11, bold=True)
    for line in [
        "доцент кафедры искусственного интеллекта",
        "Финансовый университет при Правительстве Российской Федерации, Москва, Россия",
        "pvnikitin@fa.ru; ORCID: 0000-0001-8866-5610",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        set_run_font(p.add_run(line), 10)

    abstract_ru = (
        "Актуальность. Современные методы согласования больших языковых моделей используют парные предпочтения, "
        "однако выбор между разностной и поэлементной логистическими функциями потерь часто выполняется без проверки "
        "идентифицируемости, калибровки и устойчивости к ошибкам разметки. Цель работы состоит в разработке "
        "воспроизводимого протокола сравнения этих функций при низкоранговой адаптации и в его первичной проверке "
        "на реальных предпочтительных парах. Методы. Модель Qwen2.5-3B-Instruct адаптировалась методом QLoRA на "
        "UltraFeedback. Исследованы два уровня симметричного шума меток, 0 и 0,1, и три независимых seed: 11, 29 и 47. "
        "Для каждой конфигурации использовались 256 обучающих и 64 контрольные пары, 10 шагов оптимизации и восемь "
        "фиксированных запросов для генеративной оценки. Измерялись точность ранжирования по правдоподобию, Brier score, "
        "ECE, общий сдвиг c, маржа d, дивергенция Кульбака - Лейблера к исходной SFT-модели, длина ответа, оценки спектра "
        "эмпирического Фишера и Гессиана, время и память GPU. Результаты. При отсутствии шума средняя точность составила "
        "0,5729 для попарной и 0,5677 для поэлементной потери; парная разность по совпадающим seed равна -0,0052 ± 0,0090 "
        "для контраста «поэлементная минус попарная». При шуме 0,1 соответствующие значения равны 0,5833 и 0,6042, а "
        "контраст равен 0,0208 ± 0,0325. Brier score и ECE во всех четырёх ячейках близки к 0,25 и 0,50. Генеративные "
        "показатели и спектральная оценка Гессиана характеризуются высокой межзапусковой вариативностью. Выводы. Пилот "
        "подтверждает работоспособность кода и парного протокола, но не доказывает преимущество функции потерь. Для "
        "обобщаемого вывода необходимо завершить зарегистрированный дизайн по моделям 3B, 8B и 14B, четырём наборам "
        "данных и четырём уровням шума, а также выполнить внешнюю оценку RewardBench."
    )
    add_labeled(doc, "Аннотация. ", abstract_ru)
    add_labeled(doc, "Ключевые слова: ", "обучение по предпочтениям; большие языковые модели; попарная потеря; поэлементная потеря; QLoRA; шум меток; калибровка; UltraFeedback; воспроизводимый эксперимент")

    title_en = doc.add_paragraph()
    title_en.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_en.paragraph_format.space_before = Pt(7)
    set_run_font(title_en.add_run("COMPARING PAIRWISE AND POINTWISE PREFERENCE OPTIMIZATION OF A LANGUAGE MODEL ON REAL DATA"), 12, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run("Petr V. Nikitin, PhD in Education, Associate Professor"), 10, bold=True)
    for line in [
        "Associate Professor, Department of Artificial Intelligence",
        "Financial University under the Government of the Russian Federation, Moscow, Russia",
        "pvnikitin@fa.ru; ORCID: 0000-0001-8866-5610",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        set_run_font(p.add_run(line), 9, italic=True)

    abstract_en = (
        "Background. Preference alignment of large language models relies on paired comparisons, yet pairwise and "
        "pointwise logistic objectives differ in shift identifiability, calibration, and their response to annotation "
        "errors. Objective. This study develops a reproducible protocol for comparing the two objectives under low-rank "
        "adaptation and validates the protocol on real preference data. Methods. Qwen2.5-3B-Instruct was adapted with "
        "QLoRA on UltraFeedback. Two symmetric label-noise levels, 0 and 0.1, and three independent seeds, 11, 29, and 47, "
        "were studied. Each run used 256 training pairs, 64 evaluation pairs, ten optimizer steps, and eight fixed prompts "
        "for generation-based assessment. The measurements comprised likelihood-ranking accuracy, Brier score, expected "
        "calibration error, common shift c, margin d, Kullback - Leibler divergence from the SFT reference, response length, "
        "empirical Fisher and Hessian spectral estimates, elapsed time, throughput, and peak GPU memory. Results. With no "
        "injected noise, mean ranking accuracy was 0.5729 for the pairwise objective and 0.5677 for the pointwise objective; "
        "the seed-matched pointwise-minus-pairwise contrast was -0.0052 ± 0.0090. At noise 0.1, the corresponding means were "
        "0.5833 and 0.6042, and the contrast was 0.0208 ± 0.0325. Brier score and calibration error remained close to 0.25 "
        "and 0.50 in all four cells. Generation measures and the Hessian estimate showed substantial between-seed variation. "
        "Conclusion. The pilot validates the software pipeline and the matched-seed protocol but does not establish the "
        "superiority of either loss. Generalizable evidence requires completion of the registered design across 3B, 8B, "
        "and 14B models, four datasets, four noise levels, and external RewardBench evaluation."
    )
    add_labeled(doc, "Abstract. ", abstract_en, english=True)
    add_labeled(doc, "Keywords: ", "preference learning; large language models; pairwise loss; pointwise loss; QLoRA; label noise; calibration; UltraFeedback; reproducibility", english=True)

    add_heading(doc, "ВВЕДЕНИЕ")
    add_body(doc, "Обучение по предпочтениям используется для согласования генеративных моделей с оценками человека и автоматического судьи [1 - 5]. Direct Preference Optimization преобразует сравнение двух ответов в логистическую задачу и позволяет отказаться от отдельного этапа обучения политики с подкреплением [1]. SimPO, ORPO и KTO изменяют способ нормирования и интерпретации предпочтительного сигнала [6 - 8]. Тем не менее внешне похожие функции потерь могут формировать различную геометрию оптимизации, поскольку попарный критерий наблюдает только разность оценок, а поэлементный одновременно задаёт абсолютные якоря.")
    add_body(doc, "Проблема особенно важна при параметрически эффективной адаптации. LoRA ограничивает обновление весов низкоранговым подпространством [9], QLoRA добавляет 4-битную квантизацию [10], а LoRA+, DoRA и PiSSA уточняют оптимизацию и инициализацию адаптеров [11 - 13]. При ограниченном числе обучаемых параметров конкуренция между ранжированием и абсолютным якорением может стать измеримой. Одновременно шум предпочтительных меток сжимает наблюдаемую маржу и способен менять вклад трудных примеров [14].")
    add_body(doc, "Цель исследования - сравнить попарную и поэлементную логистические потери в контролируемом факторном дизайне loss × model size × dataset × noise. На текущем этапе решаются две задачи: проверяется полный вычислительный контур на реальных данных и оценивается межзапусковая вариативность для планирования основного эксперимента. Научная новизна протокола состоит в совместном измерении качества ранжирования, вероятностной калибровки, общего сдвига, маржи, отклонения от SFT-модели, спектральной кривизны и вычислительной стоимости при совпадающих seed и совпадающих наборах пар.")
    add_body(doc, "Практическая значимость связана с выбором ответов в контакт-центрах, финансовом комплаенсе, образовательной обратной связи, справочных системах здравоохранения, промышленной поддержке, государственных сервисах и помощниках программиста. В этих областях агрегированная точность недостаточна: необходимо контролировать калибровку, безопасный отказ, изменение длины ответа, дрейф от исходной модели и категориальные ошибки на внешнем benchmark [15, 16].")

    add_heading(doc, "МАТЕМАТИЧЕСКАЯ ПОСТАНОВКА")
    add_heading(doc, "Функции потерь", 2)
    add_body(doc, "Пусть sθ(x,y) - скалярная оценка ответа y на запрос x, а yʷ и yˡ обозначают предпочтительный и отклонённый ответы. Введём a=sθ(x,yʷ), b=sθ(x,yˡ), маржу d=a-b и общий сдвиг c=(a+b)/2. Попарная логистическая потеря имеет вид")
    add_equation(doc, "L_pair(a,b)=log(1+exp(−β(a−b))).", 1)
    add_body(doc, "Поэлементная функция отдельно классифицирует два ответа:")
    add_equation(doc, "L_point(a,b)=log(1+exp(−βa))+log(1+exp(βb)).", 2)
    add_body(doc, "Попарный критерий инвариантен к преобразованию (a,b)→(a+k,b+k), поэтому направление общего сдвига не идентифицируется. Для фиксированной маржи поэлементный критерий имеет единственный минимум по c при c=0. Такое различие следует из моделей парных сравнений Брэдли - Терри и RankNet [17, 18], но при LoRA оно проявляется только в доступном адаптеру подпространстве.")
    add_heading(doc, "Шум меток", 2)
    add_body(doc, "Если истинная вероятность предпочтения первого ответа равна p, а метка независимо инвертируется с вероятностью η<0,5, наблюдаемая вероятность равна")
    add_equation(doc, "p_η=(1−η)p+η(1−p)=η+(1−2η)p.", 3)
    add_body(doc, "Шум сохраняет знак предпочтения, но приближает вероятность к 0,5. Поэтому для обоих критериев ожидается уменьшение различимости, однако конечный эффект зависит от модели, регуляризации, выборки и числа шагов.")
    add_heading(doc, "Низкоранговая адаптация", 2)
    add_body(doc, "Для исходной матрицы весов W обновление LoRA задаётся двумя матрицами малого ранга:")
    add_equation(doc, "W′=W+(α/r)BA,  rank(BA)≤r.", 4)
    add_body(doc, "В эксперименте базовые веса квантуются, а параметры A и B обучаются вместе с необходимыми выходными компонентами. Это уменьшает расход памяти и делает единый протокол применимым к моделям 3B, 8B и 14B [9 - 13].")

    add_heading(doc, "МАТЕРИАЛЫ И МЕТОДЫ")
    add_heading(doc, "Полный факторный план", 2)
    add_body(doc, "Зарегистрированный план содержит две функции потерь, три размера модели, четыре корпуса, четыре уровня шума и три seed, всего 288 запусков. Используются Qwen2.5-3B-Instruct, Qwen3-8B и Qwen2.5-14B-Instruct; UltraFeedback [19], Reddit TL;DR [3], HelpSteer2 [20] и HH-RLHF [5]; η∈{0; 0,1; 0,2; 0,3}; seed 11, 29 и 47. При статистическом анализе конкретный checkpoint учитывается отдельно от номинального размера, поскольку 8B-модель относится к другому поколению семейства Qwen [21].")
    add_table(doc, "Таблица 1. Факторы зарегистрированного эксперимента", ["Фактор", "Уровни", "Число"], [
        ["Функция потерь", "попарная; поэлементная", "2"],
        ["Модель", "Qwen2.5-3B; Qwen3-8B; Qwen2.5-14B", "3"],
        ["Данные", "UltraFeedback; Reddit TL;DR; HelpSteer2; HH-RLHF", "4"],
        ["Шум η", "0; 0,1; 0,2; 0,3", "4"],
        ["Seed", "11; 29; 47", "3"],
    ], [4.0, 10.5, 2.0])
    add_heading(doc, "Выполненный пилотный блок", 2)
    add_body(doc, "Выполнены 12 запусков для Qwen2.5-3B-Instruct на UltraFeedback: две потери, два уровня шума и три seed. Каждый запуск использовал 256 обучающих и 64 контрольные пары, максимальную длину 192 токена, 10 шагов AdamW, фактический batch size 4 за счёт накопления градиента, learning rate 2·10⁻⁴, β=0,1 и QLoRA ранга 16 при α=32 и dropout 0,05. Для генеративной проверки использовались восемь фиксированных запросов. Наборы данных формировались детерминированно; для сравнения потерь внутри seed проверено совпадение SHA-256 хешей.")
    add_heading(doc, "Метрики и воспроизводимость", 2)
    add_body(doc, "Основная метрика - likelihood-ranking accuracy, то есть доля пар, для которых нормированное логарифмическое правдоподобие предпочтительного ответа выше. Для вероятности qᵢ=σ(dᵢ) рассчитывались Brier score и expected calibration error по 15 интервалам [22, 23]:")
    add_equation(doc, "Brier=n⁻¹∑ᵢ(qᵢ−zᵢ)².", 5)
    add_equation(doc, "ECE=∑ₘ(|Bₘ|/n)|acc(Bₘ)−conf(Bₘ)|.", 6)
    add_body(doc, "Дополнительно оценивались средние c и d и их абсолютные значения, KL-дивергенция к исходной SFT-модели, win rate генераций против SFT, средняя награда, длина ответа, ведущие спектральные оценки эмпирического Фишера и Гессиана, время обучения, пропускная способность и пиковая память GPU. Произведение Гессиана на вектор вычислялось без явного формирования матрицы [24], а ведущая собственная величина оценивалась степенным методом. Полный код, конфигурация и JSON каждого запуска опубликованы в репозитории.")

    add_heading(doc, "РЕЗУЛЬТАТЫ")
    rows = []
    for loss, n, s in [("Попарная", "0", pair0), ("Поэлементная", "0", point0), ("Попарная", "0,1", pair1), ("Поэлементная", "0,1", point1)]:
        rows.append([loss, n, metric(s, "likelihood_ranking_accuracy"), metric(s, "brier"), metric(s, "ece_15"), metric(s, "kl_to_sft")])
    add_table(doc, "Таблица 2. Ранжирование, калибровка и отклонение от SFT", ["Потеря", "η", "Accuracy", "Brier", "ECE", "KL к SFT"], rows, [3.0, 1.0, 3.0, 3.0, 3.0, 3.5])
    add_body(doc, "При η=0 средняя точность попарной модели превышала поэлементную на 0,0052, однако парный контраст по трём seed мал относительно межзапусковой вариативности. При η=0,1 знак контраста изменился: поэлементная модель превосходила попарную в среднем на 0,0208, но отдельные seed дали разнонаправленные эффекты. Эти результаты нельзя трактовать как доказательство взаимодействия loss × noise: в каждой ячейке всего три запуска, а контрольная выборка содержит 64 пары.")
    add_table(doc, "Таблица 3. Парные контрасты «поэлементная минус попарная»", ["η", "Δ Accuracy", "Δ Brier", "Δ ECE", "Δ KL", "Δ win rate"], [
        ["0", delta(contrast0, "likelihood_ranking_accuracy"), delta(contrast0, "brier"), delta(contrast0, "ece_15"), delta(contrast0, "kl_to_sft"), delta(contrast0, "generation_win_rate_vs_sft")],
        ["0,1", delta(contrast1, "likelihood_ranking_accuracy"), delta(contrast1, "brier"), delta(contrast1, "ece_15"), delta(contrast1, "kl_to_sft"), delta(contrast1, "generation_win_rate_vs_sft")],
    ], [1.1, 3.0, 3.0, 3.0, 3.0, 3.4])
    add_body(doc, "Brier score во всех ячейках близок к 0,25, а ECE - к 0,50. Следовательно, после десяти шагов вероятностная шкала ещё не стала информативной, даже когда accuracy превышает случайный уровень. Это подтверждает необходимость совместного анализа ранжирования и калибровки, а не выбора checkpoint только по accuracy.")
    add_table(doc, "Таблица 4. Диагностика оценок, генерации и затрат", ["Потеря", "η", "Среднее c", "Среднее d", "Win rate", "Время, с", "GPU, GB"], [
        ["Попарная", "0", metric(pair0, "c_mean"), metric(pair0, "d_mean"), metric(pair0, "generation_win_rate_vs_sft"), metric(pair0, "train_seconds", 1), metric(pair0, "peak_gpu_memory_gb", 3)],
        ["Поэлементная", "0", metric(point0, "c_mean"), metric(point0, "d_mean"), metric(point0, "generation_win_rate_vs_sft"), metric(point0, "train_seconds", 1), metric(point0, "peak_gpu_memory_gb", 3)],
        ["Попарная", "0,1", metric(pair1, "c_mean"), metric(pair1, "d_mean"), metric(pair1, "generation_win_rate_vs_sft"), metric(pair1, "train_seconds", 1), metric(pair1, "peak_gpu_memory_gb", 3)],
        ["Поэлементная", "0,1", metric(point1, "c_mean"), metric(point1, "d_mean"), metric(point1, "generation_win_rate_vs_sft"), metric(point1, "train_seconds", 1), metric(point1, "peak_gpu_memory_gb", 3)],
    ], [2.8, 0.8, 2.7, 2.7, 2.7, 2.8, 2.8])
    add_body(doc, "Один пилотный запуск занимал около 3,2 - 3,4 минуты обучения на Tesla T4 и использовал примерно 3,94 GB пиковой памяти. Генеративный win rate оценивался только на восьми запросах на seed, поэтому его стандартное отклонение достигает 0,14 - 0,40. Ведущая оценка Гессиана оказалась численно нестабильной: изменение знака и величины между seed не позволяет использовать её для содержательного сравнения без анализа чувствительности по конечной разности и числу итераций. Оценки Фишера лучше подходят как диагностические, но также требуют большего числа пакетов.")

    add_heading(doc, "ОБСУЖДЕНИЕ")
    add_body(doc, "Полученные данные показывают, что вычислительный протокол измеряет все заявленные характеристики и сохраняет результаты отдельных запусков. При этом пилот не воспроизводит простого универсального преимущества одной функции потерь. На нулевом шуме точности почти совпадают; при η=0,1 поэлементная функция даёт положительный средний контраст, но вариативность и малый объём не позволяют отделить эффект функции от случайности оптимизации. Такой вывод согласуется с современными работами, показывающими зависимость preference optimization от предположений о полезности, регуляризации и составе данных [6 - 8, 25, 26].")
    add_body(doc, "Теоретическое различие по c в пилоте не проявилось как стабильный эмпирический разрыв. Это возможно из-за малого числа шагов, QLoRA-регуляризации и того, что абсолютный score каузальной языковой модели строится из нормированных последовательностных правдоподобий. В полном эксперименте следует заранее зафиксировать определение score, отдельно анализировать prompt-level случайные эффекты и проверять чувствительность к нормированию длины [27].")
    add_body(doc, "Для отраслевого применения набор метрик нужно связывать с ценой ошибки. В банках и страховании важны конфиденциальность, корректная эскалация и отсутствие запрещённых рекомендаций; в образовании - правильность рассуждения и педагогическая уместность; в здравоохранении - безопасный отказ и обязательный контроль специалиста; в промышленности - соблюдение регламентов; в государственных сервисах - трассируемость, равное обращение и защита данных. RewardBench, M-RewardBench и специализированные RAG-проверки могут служить внешним фильтром, но не заменяют доменную разметку [15, 16, 28, 29].")
    add_heading(doc, "Необходимые расчёты перед окончательной подачей", 2)
    add_body(doc, "Полный зарегистрированный дизайн содержит 288 запусков; выполнено 12, то есть 4,17%. Для минимальной статьи, сохраняющей исследовательский вопрос о переносимости между данными, разумно сначала завершить 3B-блок для четырёх корпусов при η∈{0;0,1}: это 48 запусков, из которых осталось 36. Для статьи с заявленным фактором размера модели необходимо выполнить все 288 запусков, то есть ещё 276, после чего построить смешанную модель с фиксированными эффектами loss, size, dataset и noise и случайным эффектом seed. RewardBench следует оценивать на финальных checkpoints по заранее установленным категориям.")
    add_body(doc, "До подачи также требуется увеличить число генеративных запросов не менее чем до 100 на конфигурацию, стабилизировать спектральную диагностику, зафиксировать правило выбора checkpoint и оценить доверительные интервалы парных контрастов. Множественные сравнения следует корректировать, а практический эффект сообщать вместе с доверительным интервалом, не ограничиваясь p-значением [30].")

    add_heading(doc, "ЗАКЛЮЧЕНИЕ")
    add_body(doc, "Разработан и проверен на реальных данных воспроизводимый контур сравнения попарной и поэлементной оптимизации предпочтений с QLoRA. Двенадцать запусков Qwen2.5-3B-Instruct на UltraFeedback подтвердили корректное вычисление accuracy, Brier, ECE, c, d, KL к SFT, генеративных показателей, спектральных оценок и вычислительных затрат. При η=0 парный контраст accuracy равен -0,0052 ± 0,0090, а при η=0,1 - 0,0208 ± 0,0325 для направления «поэлементная минус попарная». Эти значения являются пилотными и не подтверждают преимущество критерия. Главный результат текущего этапа - воспроизводимый протокол и количественная оценка вариативности, позволяющие рационально завершить факторный эксперимент без синтетических или восстановленных чисел.")

    add_heading(doc, "ДОСТУПНОСТЬ КОДА И ДАННЫХ")
    add_body(doc, "Код, notebook Colab, манифест 288 запусков, JSON-результаты и агрегаты опубликованы в репозитории https://github.com/PetrNikitin20/optimML. Сырые тексты корпусов не перераспространяются; они загружаются из первичных источников в соответствии с их условиями использования.")

    add_heading(doc, "СПИСОК ЛИТЕРАТУРЫ")
    references = [
        "Rafailov R., Sharma A., Mitchell E., Manning C.D., Ermon S., Finn C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems. 2023. Vol. 36. P. 53728-53741.",
        "Ouyang L. et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems. 2022. Vol. 35. P. 27730-27744.",
        "Stiennon N. et al. Learning to summarize with human feedback. Advances in Neural Information Processing Systems. 2020. Vol. 33. P. 3008-3021.",
        "Christiano P.F. et al. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems. 2017. Vol. 30.",
        "Bai Y. et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv:2204.05862. 2022.",
        "Meng Y., Xia M., Chen D. SimPO: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems. 2024. Vol. 37. DOI: 10.52202/079017-3946.",
        "Hong J., Lee N., Thorne J. ORPO: Monolithic preference optimization without reference model. Proceedings of EMNLP. 2024. P. 11170-11189. DOI: 10.18653/v1/2024.emnlp-main.626.",
        "Ethayarajh K., Xu W., Muennighoff N., Jurafsky D., Kiela D. Model alignment as prospect theoretic optimization. Proceedings of ICML. 2024. Vol. 235. P. 12634-12651.",
        "Hu E.J. et al. LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations. 2022. URL: https://openreview.net/forum?id=nZeVKeeFYf9.",
        "Dettmers T., Pagnoni A., Holtzman A., Zettlemoyer L. QLoRA: Efficient finetuning of quantized LLMs. Advances in Neural Information Processing Systems. 2023. Vol. 36. P. 10088-10115.",
        "Hayou S., Ghosh N., Yu B. LoRA+: Efficient low rank adaptation of large models. Proceedings of ICML. 2024. Vol. 235. P. 17783-17806.",
        "Liu S.-Y. et al. DoRA: Weight-decomposed low-rank adaptation. Proceedings of ICML. 2024. Vol. 235. P. 32100-32121.",
        "Meng F., Wang Z., Zhang M. PiSSA: Principal singular values and singular vectors adaptation of large language models. Advances in Neural Information Processing Systems. 2024. Vol. 37.",
        "Natarajan N., Dhillon I.S., Ravikumar P.K., Tewari A. Learning with noisy labels. Advances in Neural Information Processing Systems. 2013. Vol. 26.",
        "Lambert N. et al. RewardBench: Evaluating reward models for language modeling. Findings of NAACL. 2025. P. 1755-1797. DOI: 10.18653/v1/2025.findings-naacl.96.",
        "Gureja S. et al. M-RewardBench: Evaluating reward models in multilingual settings. Proceedings of ACL. 2025. P. 43-58. DOI: 10.18653/v1/2025.acl-long.3.",
        "Bradley R.A., Terry M.E. Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika. 1952. Vol. 39. No. 3/4. P. 324-345. DOI: 10.1093/biomet/39.3-4.324.",
        "Burges C. et al. Learning to rank using gradient descent. Proceedings of ICML. 2005. P. 89-96. DOI: 10.1145/1102351.1102363.",
        "Cui G. et al. UltraFeedback: Boosting language models with scaled AI feedback. arXiv:2310.01377. 2023.",
        "Wang Z. et al. HelpSteer2: Open-source dataset for training top-performing reward models. arXiv:2406.08673. 2024.",
        "Yang A. et al. Qwen2.5 technical report. arXiv:2412.15115. 2024.",
        "Brier G.W. Verification of forecasts expressed in terms of probability. Monthly Weather Review. 1950. Vol. 78. No. 1. P. 1-3.",
        "Guo C., Pleiss G., Sun Y., Weinberger K.Q. On calibration of modern neural networks. Proceedings of ICML. 2017. Vol. 70. P. 1321-1330.",
        "Pearlmutter B.A. Fast exact multiplication by the Hessian. Neural Computation. 1994. Vol. 6. No. 1. P. 147-160. DOI: 10.1162/neco.1994.6.1.147.",
        "Chen A. et al. Preference learning algorithms do not learn preference rankings. Advances in Neural Information Processing Systems. 2024. Vol. 37.",
        "Azar M.G. et al. A general theoretical paradigm to understand learning from human preferences. arXiv:2310.12036. 2023.",
        "Dubois Y., Galambosi B., Liang P., Hashimoto T.B. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. arXiv:2404.04475. 2024.",
        "Jin Z. et al. RAG-RewardBench: Benchmarking reward models in retrieval augmented generation for preference alignment. Findings of ACL. 2025. P. 17061-17090. DOI: 10.18653/v1/2025.findings-acl.877.",
        "Ethayarajh K., Xu W., Muennighoff N., Jurafsky D., Kiela D. Understanding dataset difficulty with V-usable information. Proceedings of ICML. 2022. Vol. 162. P. 5988-6008.",
        "Benjamini Y., Hochberg Y. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society. Series B. 1995. Vol. 57. No. 1. P. 289-300.",
    ]
    for idx, reference in enumerate(references, 1):
        add_reference(doc, idx, reference)

    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    doc.core_properties.title = "Сравнение попарной и поэлементной оптимизации предпочтений языковой модели на реальных данных"
    doc.core_properties.author = "Никитин Петр Владимирович"
    doc.core_properties.last_modified_by = "Никитин Петр Владимирович"
    doc.core_properties.subject = "Статья для журнала «Программные продукты и системы»"
    doc.core_properties.keywords = "preference optimization, QLoRA, label noise, calibration, UltraFeedback"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
