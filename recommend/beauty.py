# main.py
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

BASE_URL = "https://api.musinsa.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.musinsa.com/",
}


# -----------------------------------
# 1) PLP API 호출 함수
# -----------------------------------
def fetch_beauty_plp(
    category: str = "104015",
    page: int = 1,
    size: int = 60,
    color: Optional[str] = None,
    sort_code: str = "POPULAR",
) -> Dict[str, Any]:
    """
    무신사 뷰티 카테고리 PLP JSON 호출
    예: https://api.musinsa.com/api2/dp/v1/plp/goods?...&category=104015&page=1
    """
    url = f"{BASE_URL}/api2/dp/v1/plp/goods"

    params: Dict[str, Any] = {
        "gf": "A",
        "category": category,
        "page": page,
        "size": size,
        "sortCode": sort_code,
        "caller": "CATEGORY",
        "seen": 0,
        "seenAds": "",
        "testGroup": "",
    }
    if color:
        params["color"] = color

    resp = requests.get(url, headers=HEADERS, params=params, timeout=8)
    resp.raise_for_status()

    try:
        return resp.json()
    except Exception:
        print("❌ JSON 파싱 실패:", resp.text[:500])
        raise


# -----------------------------------
# 2) PLP JSON → 우리 쪽 상품 구조로 변환
# -----------------------------------
def extract_products_from_plp(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    PLP 응답에서 goods 리스트만 뽑아서, 필요한 필드만 매핑.
    """
    products: List[Dict[str, Any]] = []

    # 래핑 구조 방어적으로 처리
    # 실제 구조가 {"data": {"goods": [...]}} 면 첫 줄이 동작
    goods_list = (
        data.get("data", {}).get("list")
        or data.get("list")
        or (data if isinstance(data, list) else [])
    )

    if not goods_list:
        return products

    for g in goods_list:
        goods_no = g.get("goodsNo")
        goods_name = g.get("goodsName")
        goods_link_url = g.get("goodsLinkUrl")

        thumbnail = g.get("thumbnail")
        display_gender = g.get("displayGenderText")

        normal_price = g.get("normalPrice")
        price = g.get("price")
        coupon_price = g.get("couponPrice")
        sale_rate = g.get("saleRate")
        coupon_sale_rate = g.get("couponSaleRate")

        brand = g.get("brand")
        brand_name = g.get("brandName")
        brand_link = g.get("brandLinkUrl")

        is_sold_out = g.get("isSoldOut")
        review_count = g.get("reviewCount")
        review_score = g.get("reviewScore")
        is_option_visible = g.get("isOptionVisible")

        product = {
            "product_id": goods_no,
            "name": goods_name,
            "product_url": goods_link_url,
            "image_url": thumbnail,

            "final_price": price,
            "original_price": normal_price,
            "coupon_price": coupon_price,
            "discount_rate": sale_rate,
            "coupon_discount_rate": coupon_sale_rate,

            "brand": brand,
            "brand_name": brand_name,
            "brand_url": brand_link,

            "gender": display_gender,
            "sold_out": is_sold_out,
            "review_count": review_count,
            "review_score": review_score,
            "is_option_visible": is_option_visible,

            # 이 API에는 옵션 이름이 아예 안 실려 있음
            "options": [],
        }

        products.append(product)

    return products


