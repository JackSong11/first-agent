import fitz  # PyMuPDF


def identify_pdf_type(file_path, threshold=10):
    """
    判断 PDF 是扫描件还是电子档
    :param file_path: PDF 文件路径
    :param threshold: 判定为电子档的最小字符数阈值
    :return: 'scanned' (扫描件), 'digital' (电子档), 'searchable_scanned' (带OCR层的扫描件)
    """
    doc = fitz.open(file_path)

    # 为了提高效率，通常检查前 3-5 页即可
    check_pages = min(len(doc), 5)
    is_scanned_results = []

    for i in range(check_pages):
        page = doc[i]

        # 1. 提取页面文本
        text = page.get_text().strip()
        # 2. 提取页面图像列表
        image_list = page.get_images(full=True)

        # 逻辑判断
        if len(text) < threshold:
            # 几乎没有文字，且含有图片 -> 确认为扫描件
            if len(image_list) > 0:
                is_scanned_results.append('scanned')
            else:
                # 既无文字也无图片（可能是空白页或纯矢量图形）
                is_scanned_results.append('scanned')
        else:
            # 文字较多
            if len(image_list) > 0:
                # 既有大量文字又有图片，可能是带 OCR 层的扫描件
                # 也可以直接视为 digital 处理（因为可以直接提取文本）
                is_scanned_results.append('digital')
            else:
                is_scanned_results.append('digital')

    doc.close()

    # 简单统计多数决策
    scanned_count = is_scanned_results.count('scanned')
    if scanned_count > (check_pages / 2):
        return "scanned"
    else:
        return "digital"


# --- 使用示例 ---
pdf_path = "/Users/songzhiquan1/Documents/test_opencode/1765249283082-2151892715866947584.pdf"
pdf_type = identify_pdf_type(pdf_path)

if pdf_type == "scanned":
    print(f"检测结果: 扫描件 -> 建议调用 OCR 大模型 (如 Qwen-VL / LayoutLM)")
else:
    print(f"检测结果: 原生电子档 -> 建议直接提取文本流 (效率更高)")