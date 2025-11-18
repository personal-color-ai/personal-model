# olive_best_html.py

from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, parse_qs

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

BASE_URL = "https://www.oliveyoung.co.kr"


def _extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def scrape_olive_best_html(
    disp_cat_no: str,
    headless: bool = True,
) -> List[Dict[str, Any]]:
    """
    올리브영 베스트 페이지 HTML을 그대로 렌더링해서,
    div.TabsConts.on 안에 있는 상품 li들을 파싱해서 JSON 리스트로 리턴.
    """

    url = f"{BASE_URL}/store/main/getBestList.do?dispCatNo={disp_cat_no}"

    # 1) Playwright로 실제 브라우저 띄워서 HTML 가져오기
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()

    # 2) BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(html, "html.parser")

    tabs_cont = soup.select_one("div.TabsConts.on")
    if not tabs_cont:
        raise RuntimeError('div.TabsConts.on 을 찾지 못했습니다.')

    # 실제 구조 보고 필요하면 여기 selector만 손보면 됨
    # 일단 li 전부 대상으로 잡고, 안에 필요한 태그만 골라서 씀
    items = tabs_cont.select("li")
    products: List[Dict[str, Any]] = []

    for li in items:
        # 이름 / 브랜드 / 가격 후보 selector 여러 개 써놓기
        name_tag = li.select_one(".prd_name, .tx_name, .goods_name, .name")
        brand_tag = li.select_one(".tx_brand, .brand, .brandNm")
        price_tag = li.select_one(".tx_cur, .tx_price, .price")

        link_tag = li.select_one("a")
        img_tag = li.select_one("img")

        name = name_tag.get_text(strip=True) if name_tag else None
        if not name:
            # 이름도 없는 li는 그냥 스킵
            continue

        brand = brand_tag.get_text(strip=True) if brand_tag else None
        price_raw = price_tag.get_text(strip=True) if price_tag else None
        price = _extract_int(price_raw)

        # 상품 상세 링크
        href = link_tag["href"] if link_tag and link_tag.has_attr("href") else None
        product_url = urljoin(BASE_URL, href) if href else None

        goods_no = None
        if href:
            try:
                parsed = urlparse(href)
                qs = parse_qs(parsed.query)
                goods_no = qs.get("goodsNo", [None])[0]
            except Exception:
                pass

        # 이미지 URL
        img_src = None
        if img_tag:
            for attr in ("data-original", "data-src", "src"):
                if img_tag.has_attr(attr) and img_tag[attr]:
                    img_src = img_tag[attr]
                    break

        if img_src:
            if img_src.startswith("//"):
                img_src = "https:" + img_src
            elif img_src.startswith("/"):
                img_src = urljoin(BASE_URL, img_src)

        products.append(
            {
                "goods_no": goods_no,
                "name": name,
                "brand": brand,
                "price": price,
                "price_raw": price_raw,
                "product_url": product_url,
                "image_url": img_src,
            }
        )

    return products
