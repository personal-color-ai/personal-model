from typing import Dict, Any, List, Optional

import requests

# ------------------------------
# 무신사 - 뷰티 - 상품별 리뷰 크롤러
# ------------------------------

REVIEW_BASE_URL = "https://goods.musinsa.com"

REVIEW_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.musinsa.com/",
}


def fetch_picture_reviews_json(
    goods_no: int | str,
    page: int = 1,
    size: int = 20,
) -> Dict[str, Any]:
    """
    무신사 포토 리뷰 JSON 호출
    예:
    https://goods.musinsa.com/api2/review/v1/picture-reviews?goodsNo=3882622&size=20&page=1
    """
    url = f"{REVIEW_BASE_URL}/api2/review/v1/picture-reviews"
    params = {
        "goodsNo": goods_no,
        "page": page,
        "size": size,
    }

    resp = requests.get(url, headers=REVIEW_HEADERS, params=params, timeout=10)
    resp.raise_for_status()

    try:
        return resp.json()
    except Exception:
        print("❌ 리뷰 JSON 파싱 실패:", resp.text[:500])
        raise


def _full_image_url(path: Optional[str]) -> Optional[str]:
    """
    리뷰 이미지/상품 썸네일이 '/data/estimate/...' 처럼 오는 경우
    image.msscdn.net 기준으로 풀 URL로 바꿔주는 헬퍼.
    """
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    # 무신사 이미지 CDN 기본 도메인
    return f"https://image.msscdn.net{path}"


def extract_picture_reviews(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    picture-reviews 응답(JSON)에서 우리가 쓰기 편한 구조로 변환.
    """
    data_root = data.get("data", {}) or {}

    total_count = data_root.get("totalCount", 0)
    raw_list = data_root.get("list", []) or []

    reviews: List[Dict[str, Any]] = []

    for r in raw_list:
        goods_info = r.get("goods") or {}

        images_raw = r.get("images") or []
        images = [
            {
                "image_no": img.get("imageNo"),
                "alt": img.get("altText"),
                "url": _full_image_url(img.get("image")),
            }
            for img in images_raw
        ]

        review = {
            "review_id": r.get("no"),
            "type": r.get("type"),
            "grade": r.get("grade"),
            "content": r.get("content"),
            "comment_count": r.get("commentCount"),
            "like_count": r.get("likeCount"),

            # 리뷰 작성자 정보
            "user_name": r.get("userProfileName"),
            "user_profile": r.get("userProfile"),
            "user_image": _full_image_url(r.get("userImageFile")),
            "has_skin_worry": r.get("hasSkinWorry"),
            "show_user_profile": r.get("showUserProfile"),

            # 상품 정보 (리뷰 안에 포함된 요약)
            "goods_no": goods_info.get("goodsNo"),
            "goods_name": goods_info.get("goodsName"),
            "goods_option": r.get("goodsOption"),
            "goods_brand": goods_info.get("brandName"),
            "goods_thumbnail": _full_image_url(
                goods_info.get("goodsThumbnailImageUrl")
                or goods_info.get("goodsImageFile")
            ),

            # 이미지 리스트
            "images": images,

            # 기타 메타
            "created_at": r.get("pastDate"),
            "experience": r.get("experience"),
            "my_review": r.get("myReview"),
        }

        reviews.append(review)

    return {
        "total_count": total_count,
        "review_count": len(reviews),
        "reviews": reviews,
    }
