from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException

# ------------------------------
# 무신사 - 패션 카테고리별 랭킹 크롤러
# ------------------------------

BASE_URL = "https://api.musinsa.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.musinsa.com/main/musinsa/ranking",
}


# ------------------------------
# 1. 랭킹 설정(JSON) 가져오기
# ------------------------------
def get_ranking_config(
    store_code: str = "musinsa", sub_pan: str = "product"
) -> Dict[str, Any]:
    """
    무신사 랭킹판 설정 JSON
    예: https://api.musinsa.com/api2/hm/web/v5/pans/ranking?storeCode=musinsa&subPan=product
    """
    url = f"{BASE_URL}/api2/hm/web/v5/pans/ranking"
    params = {
        "storeCode": store_code,
        "subPan": sub_pan,
    }
    resp = requests.get(url, headers=HEADERS, params=params, timeout=5)
    resp.raise_for_status()

    try:
        return resp.json()
    except Exception:
        # 디버깅용: 응답이 json이 아닐 때 확인
        print("config not json, body snippet:", resp.text[:500])
        raise


def find_tab_outlined_module(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    modules 중에서 type == 'TAB_OUTLINED' 인 모듈 하나 찾기
    (카테고리 탭들 + apiUrl 템플릿이 들어있음)
    """
    for m in config.get("data").get("modules", []):
        if m.get("type") == "TAB_OUTLINED":
            return m
    return None


# ------------------------------
# 2. 카테고리 목록 뽑기
# ------------------------------
def extract_all_categories(
    tab_module: Dict[str, Any]
) -> List[Dict[str, Optional[str]]]:
    """
    TAB_OUTLINED 모듈에서 모든 (depth1, depth2, sectionId, categoryCode) 조합 추출.
    depth2가 없는 경우도 같이 포함.
    반환 예:
    [
      {
        "depth1": "상의",
        "depth2": None,
        "section_id": "200",
        "category_code": "001",
      },
      {
        "depth1": "상의",
        "depth2": "셔츠/블라우스",
        "section_id": "201",
        "category_code": "001002",
      },
      ...
    ]
    """
    categories: List[Dict[str, Optional[str]]] = []

    tabs_lv1 = tab_module.get("tabs", [])
    for t1 in tabs_lv1:
        t1_text = (t1.get("text") or {}).get("text")
        t1_params = t1.get("params") or {}
        t1_section_id = t1_params.get("sectionId")
        t1_category_code = t1_params.get("categoryCode")

        # 1뎁스 전체 카테고리 (depth2 없음)
        if t1_section_id and t1_category_code:
            categories.append(
                {
                    "depth1": t1_text,
                    "depth2": None,
                    "section_id": t1_section_id,
                    "category_code": t1_category_code,
                }
            )

        # 2뎁스 탭들
        for t2 in t1.get("tabs", []) or []:
            t2_text = (t2.get("text") or {}).get("text")
            t2_params = t2.get("params") or {}
            t2_section_id = t2_params.get("sectionId")
            t2_category_code = t2_params.get("categoryCode")

            if t2_section_id and t2_category_code:
                categories.append(
                    {
                        "depth1": t1_text,
                        "depth2": t2_text,
                        "section_id": t2_section_id,
                        "category_code": t2_category_code,
                    }
                )

    # 중복 제거 (sectionId + categoryCode 기준)
    seen = set()
    uniq: List[Dict[str, Optional[str]]] = []
    for c in categories:
        key = (c["section_id"], c["category_code"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    return uniq


def build_section_url(
    api_url_template: str,
    section_id: str,
    category_code: str,
    contents_id: str = "",
) -> str:
    """
    TAB_OUTLINED.apiUrl 템플릿에 sectionId / categoryCode / contentsId 채워서
    실제 호출할 URL path 만들기.
    예: /api2/hm/web/v5/pans/ranking/sections/200?storeCode=musinsa&categoryCode=001002&contentsId=
    """
    return api_url_template.format(
        sectionId=section_id,
        categoryCode=category_code,
        contentsId=contents_id or "",
    )


def fetch_section(section_url: str) -> Dict[str, Any]:
    """
    섹션(카테고리) 랭킹 데이터 JSON 가져오기.
    section_url 이 / 로 시작하면 BASE_URL 붙여줌.
    """
    if section_url.startswith("http"):
        url = section_url
    else:
        url = BASE_URL + section_url

    resp = requests.get(url, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        print("section not json, url:", url)
        print(resp.text[:500])
        raise


# ------------------------------
# 3. 섹션 JSON에서 상품 리스트만 뽑기
# ------------------------------
def extract_products_from_section(data: dict) -> list[dict]:
    """
    무신사 MULTICOLUMN / PRODUCT_COLUMN 구조에서
    최소한의 상품 정보만 뽑아서 리스트로 리턴.

    반환 예시:
    [
      {
        "product_id": "5620937",
        "brand": "노운",
        "name": "washed biker cargo pants (gray)",
        "final_price": 106560,
        "original_price": 148000,
        "discount_rate": 28,
        "rank": 96,
        "image_url": "https://image.msscdn.net/...",
        "product_url": "https://www.musinsa.com/products/5620937",
        "watching_text": "63명이 보는 중",
    ...
      }
    ]
    """

    products: list[dict] = []

    # 1) data 최상단에 modules 가 있는 경우 (섹션 전체 JSON)
    modules = data.get("data").get("modules")
    # 2) 지금 보여준 것처럼 data 자체가 MULTICOLUMN 하나인 경우도 처리
    if modules is None and data.get("type") == "MULTICOLUMN":
        modules = [data]

    if not modules:
        return products

    for module in modules:
        if module.get("type") != "MULTICOLUMN":
            continue

        for col in module.get("items", []):
            if col.get("type") != "PRODUCT_COLUMN":
                continue

            info = col.get("info", {}) or {}
            image = col.get("image", {}) or {}
            on_click = col.get("onClick", {}) or {}

            # like / impression 로그 쪽 payload (원가 / 베스트가 들어있음)
            like_payload = (
                (
                    (image.get("onClickLike") or {})
                    .get("eventLog", {})
                    .get("ga4", {})
                    .get("payload", {})
                )
                or {}
            )
            impression_payload = (
                (
                    (col.get("impressionEventLog") or {})
                    .get("ga4", {})
                    .get("payload", {})
                )
                or {}
            )

            # 가격 정보
            final_price = info.get("finalPrice")
            # original_price, best_price는 로그에서 가져옴 (문자/숫자 혼합이라 캐스팅)
            def _to_int(v):
                try:
                    return int(v)
                except Exception:
                    return None

            original_price = (
                _to_int(like_payload.get("original_price"))
                or _to_int(impression_payload.get("original_price"))
            )
            best_price = (
                _to_int(like_payload.get("best_price"))
                or _to_int(impression_payload.get("best_price"))
                or final_price
            )

            # “63명이 보는 중” 같은 추가 정보
            watching_text = None
            add_info = info.get("additionalInformation") or []
            if add_info and isinstance(add_info, list):
                first = add_info[0] or {}
                watching_text = first.get("text")

            product = {
                "product_id": col.get("id"),
                "brand": info.get("brandName"),
                "name": info.get("productName"),

                "final_price": final_price,
                "original_price": original_price,
                "best_price": best_price,
                "discount_rate": info.get("discountRatio"),

                "rank": image.get("rank"),
                "image_url": image.get("url"),
                "product_url": on_click.get("url"),

                "watching_text": watching_text,
            }

            products.append(product)

    return products



# ------------------------------
# 4. 카테고리별 랭킹 조회
# ------------------------------
def get_all_category_ranking(
    store_code: str = "musinsa",
    sub_pan: str = "product",
    limit_per_category: int = 20,
) -> Dict[str, Any]:
    """
    1) 설정 JSON 가져오기
    2) TAB_OUTLINED 모듈 찾기
    3) 모든 카테고리(depth1/depth2) 리스트 뽑기
    4) 카테고리별 섹션 랭킹을 호출하고 최대 limit_per_category개씩만 담아서 반환
    """
    config = get_ranking_config(store_code=store_code, sub_pan=sub_pan)

    tab_module = find_tab_outlined_module(config)
    if not tab_module:
        raise RuntimeError("TAB_OUTLINED 모듈을 찾을 수 없습니다.")

    api_url_template = tab_module["apiUrl"]
    categories = extract_all_categories(tab_module)

    result_categories: List[Dict[str, Any]] = []

    for c in categories:
        section_id = c["section_id"]
        category_code = c["category_code"]
        depth1 = c["depth1"]
        depth2 = c["depth2"]

        # 섹션 URL 만들기
        section_url = build_section_url(
            api_url_template, section_id=section_id, category_code=category_code
        )

        try:
            section_data = fetch_section(section_url)
            items = extract_products_from_section(section_data)
        except Exception as e:
            # 카테고리 하나 오류 나도 전체가 깨지지 않게
            print(
                f"[WARN] 카테고리 {depth1}/{depth2} 섹션 조회 실패:", repr(e)
            )
            items = []

        result_categories.append(
            {
                "depth1": depth1,
                "depth2": depth2,
                "section_id": section_id,
                "category_code": category_code,
                "item_count": len(items),
                "items": items[:limit_per_category],
            }
        )

    return {
        "store_code": store_code,
        "category_count": len(result_categories),
        "categories": result_categories,
    }


