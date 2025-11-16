from typing import Optional

import requests
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from recommend.beauty import extract_products_from_plp, fetch_beauty_plp
from recommend.musinsa import get_all_category_ranking, extract_all_categories, find_tab_outlined_module, \
    get_ranking_config
from recommend.option import fetch_goods_detail_options_json, extract_goods_options
from recommend.review import fetch_picture_reviews_json, extract_picture_reviews
from test import scrape_olive_best_html


router = APIRouter(prefix="/crawling", tags=["crawling"])


@router.post("/musinsa/categories")
def musinsa_categories(store_code: str = "musinsa"):
    """
    현재 랭킹판에서 어떤 카테고리(depth1/depth2)가 있는지 목록만 보고 싶을 때.

    예:
      GET /musinsa/categories
    """
    try:
        config = get_ranking_config(store_code=store_code, sub_pan="product")
        tab_module = find_tab_outlined_module(config)
        if not tab_module:
            raise RuntimeError("TAB_OUTLINED 모듈을 찾을 수 없습니다.")

        categories = extract_all_categories(tab_module)
        return {
            "store_code": store_code,
            "category_count": len(categories),
            "categories": categories,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("musinsa categories error:", repr(e))
        raise HTTPException(status_code=500, detail="카테고리 조회 중 오류가 발생했습니다.")


@router.post("/musinsa/fashion")
def musinsa_ranking_all(
    store_code: str = "musinsa",
    sub_pan: str = "product",
    limit_per_category: int = 20,
):
    """
    모든 카테고리를 for문으로 돌면서 카테고리별 랭킹 상품들을 가져오는 엔드포인트.

    예:
      GET /musinsa/ranking/all
      GET /musinsa/ranking/all?limit_per_category=5
    """
    try:
        result = get_all_category_ranking(
            store_code=store_code,
            sub_pan=sub_pan,
            limit_per_category=limit_per_category,
        )
        return result.get("categories").get(0).get("items", [])
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("musinsa ranking all error:", repr(e))
        raise HTTPException(status_code=500, detail="랭킹 조회 중 오류가 발생했습니다.")


@router.post("/musinsa/beauty")
def get_musinsa_beauty(
    category: str = "104015",
    page: int = 1,
    size: int = 60,
    color: Optional[str] = None,
    sort_code: str = "POPULAR",
):
    """
    예: /musinsa/beauty?category=104015&page=1&size=60
    → PLP 기반 뷰티 상품 리스트 JSON 반환
    """
    try:
        raw = fetch_beauty_plp(
            category=category,
            page=page,
            size=size,
            color=color,
            sort_code=sort_code,
        )
        items = extract_products_from_plp(raw)
    except requests.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"무신사 API 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")

    # TODO 데이터 응답 구조 없이 리스틑만 반환
    data = {
        "category": category,
        "page": page,
        "size": size,
        "color": color,
        "sort_code": sort_code,
        "count": len(items),
        "items": items,
    }

    return items

@router.post("/musinsa/reviews")
def get_picture_reviews(
    goods_no: int,
    page: int = 1,
    size: int = 20,
):
    """
    예:
    /musinsa/reviews/pictures?goods_no=3882622&page=1&size=20
    """
    try:
        raw = fetch_picture_reviews_json(goods_no=goods_no, page=page, size=size)
        parsed = extract_picture_reviews(raw)
    except requests.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"무신사 리뷰 API 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")

    data = {
        "goods_no": goods_no,
        "page": page,
        "size": size,
        "total_count": parsed["total_count"],
        "count": parsed["review_count"],
        "reviews": parsed["reviews"],
    }

    return parsed["reviews"]

@router.post("/musinsa/options")
def get_option(
    goods_no: int
):
    """ 상품 상세 조회 - 옵션 불러오기
    https://goods-detail.musinsa.com/api2/goods/2396304/options?goodsSaleType=SALE&optKindCd=BEAUTY
    """
    try:
        raw = fetch_goods_detail_options_json(goods_no=goods_no)
        parsed = extract_goods_options(raw)
    except requests.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"무신사 옵션 API 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")

    return parsed["options"]



@router.post("/oliveyoung/best")
def oliveyoung_best(
    disp_cat_no: str = "900000100100001",
    limit: int = 20,
):
    """
    올리브영 베스트 상품 랭킹 조회 API

    예:
    GET /oliveyoung/best?disp_cat_no=900000100100001&limit=10
    """
    try:

        disp_no = "900000100100001"

        items = scrape_olive_best_html(disp_cat_no=disp_no, headless=False)

        print("상품 개수:", len(items))

        return JSONResponse(content=items)

    except HTTPException:
        # 이미 위에서 HTTPException으로 던진 경우 그대로 전달
        raise
    except Exception as e:
        print("[ERROR] /oliveyoung/best 실패:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="올리브영 베스트 랭킹 조회 중 오류가 발생했습니다.",
        )