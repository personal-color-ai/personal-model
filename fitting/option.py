import requests
from typing import Dict, Any, List, Optional

GOODS_DETAIL_BASE_URL = "https://goods-detail.musinsa.com"

GOODS_DETAIL_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.musinsa.com/",
}


def fetch_goods_detail_options_json(
    goods_no: int | str,
    goods_sale_type: str = "SALE",
    opt_kind_cd: str = "BEAUTY",
) -> Dict[str, Any]:
    """
    무신사 상품 상세 옵션 JSON 호출
    예:
    https://goods-detail.musinsa.com/api2/goods/2396304/options?goodsSaleType=SALE&optKindCd=BEAUTY
    """
    url = f"{GOODS_DETAIL_BASE_URL}/api2/goods/{goods_no}/options"

    params = {
        "goodsSaleType": goods_sale_type,
        "optKindCd": opt_kind_cd,
    }

    resp = requests.get(url, headers=GOODS_DETAIL_HEADERS, params=params, timeout=10)
    resp.raise_for_status()

    try:
        return resp.json()
    except Exception:
        print("❌ 옵션 JSON 파싱 실패:", resp.text[:500])
        raise


def extract_goods_options(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    goods-detail 옵션 응답(JSON)을 우리가 쓰기 편한 구조로 변환.

    - group 단위(basic 배열의 각 항목)
    - group 안의 optionValues를 하나의 리스트로 flatten
    """
    data_root = data.get("data", {}) or {}
    basic_list = data_root.get("basic") or []

    groups: List[Dict[str, Any]] = []
    flat_options: List[Dict[str, Any]] = []

    for group in basic_list:
        group_no = group.get("no")
        group_name = group.get("name")     # 예: "C"
        group_type = group.get("type")     # "basic"
        display_type = group.get("displayType")  # "DROPDOWN" 등

        option_values = group.get("optionValues") or []

        group_options: List[Dict[str, Any]] = []

        for ov in option_values:
            color = ov.get("color") or {}
            opt = {
                "value_no": ov.get("no"),
                "option_no": ov.get("optionNo"),
                "name": ov.get("name"),          # 예: "002 손웜수템"
                "code": ov.get("code"),          # 예: "002 손웜수템"
                "sequence": ov.get("sequence"),
                "color_code": color.get("colorCode"),
                "color_type": color.get("colorType"),
                "image_url": ov.get("imageUrl"),
                "is_deleted": ov.get("isDeleted"),
                "is_leaf": ov.get("isLeaf"),
                "out_of_stock": ov.get("outOfStock"),
            }

            group_options.append(opt)

            # 전체 flat 리스트에도 추가
            flat_options.append(
                {
                    "group_no": group_no,
                    "group_name": group_name,
                    "group_type": group_type,
                    "display_type": display_type,
                    **opt,
                }
            )

        groups.append(
            {
                "group_no": group_no,
                "group_name": group_name,
                "group_type": group_type,
                "display_type": display_type,
                "options": group_options,
            }
        )

    return {
        "groups": groups,          # 옵션 그룹별 구조
        "options": flat_options,   # group 정보까지 포함된 flat 리스트
    }
