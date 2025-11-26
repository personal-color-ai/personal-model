from typing import List
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import chromadb

router = APIRouter(prefix="/embedding", tags=["embedding"])

# ---------------------------------------------------------
# ChromaDB 클라이언트 설정
# ---------------------------------------------------------
# path="./chroma_db"는 프로젝트 루트의 해당 폴더에 DB 파일을 저장
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="review_collection")


# ---------------------------------------------------------
# Request Body 모델 (Pydantic)
# ---------------------------------------------------------
class ReviewItem(BaseModel):
    rating: int
    likes: int
    content: str
    userDescription: str  # 예: "지성 · 가을 웜톤 · 모공, 여드름"


class ProductReviewPayload(BaseModel):
    id: int
    name: str
    brand: str
    rating: int
    reviewCountAll: int
    category: str
    review: List[ReviewItem]

# 2. 검색(조회)용 모델
class SearchRequest(BaseModel):
    personal_color: str  # ex: "여름" -> 메타데이터 필터링용
    prompt: str          # ex: "지속력 좋은 틴트 추천해줘" -> 벡터 유사도 검색용


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@router.post("")
def embed_product(payloads: List[ProductReviewPayload]):
    """
    여러 상품의 정보와 리뷰 데이터를 리스트 형태로 받아 ChromaDB에 저장

    [입력 예시]
    [
      { "id": 11, "name": "상품A", "review": [...] },
      { "id": 12, "name": "상품B", "review": [...] }
    ]
    """
    if not payloads:
        return {"message": "저장할 데이터가 없습니다."}

    total_embedded_count = 0
    total_products = 0

    # 배치 처리를 위한 리스트 초기화
    all_ids = []
    all_documents = []
    all_metadatas = []

    # 1. 전달받은 모든 상품(payloads)을 순회
    for product in payloads:
        if not product.review:
            continue

        total_products += 1

        # 2. 각 상품의 리뷰들을 순회하며 데이터 준비
        for i, item in enumerate(product.review):
            doc_text = f"상품명: {product.name}\n브랜드: {product.brand}\n리뷰내용: {item.content}"

            meta = {
                "product_id": product.id,
                "product_name": product.name,
                "brand": product.brand,
                "category": product.category,  # [핵심] 카테고리 정보 메타데이터에 저장
                "user_description": item.userDescription,
                "review_content": item.content,
                "review_rating": item.rating
            }

            # ID 생성: 중복 방지를 위해 상품ID + 인덱스 + UUID 조합
            unique_id = f"prod_{product.id}_rev_{i}_{uuid.uuid4().hex[:8]}"

            all_ids.append(unique_id)
            all_documents.append(doc_text)
            all_metadatas.append(meta)

    # 데이터가 없으면 종료
    if not all_ids:
        return {"message": "임베딩할 데이터가 없습니다."}

    try:
        # 3. ChromaDB에 한 번에 저장 (Batch Upsert)
        # 데이터 양이 매우 많다면(수천 개 이상) 여기서 chunk 단위로 끊어서 넣는 로직이 추가로 필요
        collection.upsert(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas
        )
        total_embedded_count = len(all_ids)

    except Exception as e:
        print(f"[Error] ChromaDB Embedding Failed: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 저장 실패: {str(e)}")

    return {
        "message": "성공적으로 임베딩되었습니다.",
        "processed_products_count": total_products,
        "total_embedded_reviews": total_embedded_count
    }


@router.post("/search")
def search_reviews(request: SearchRequest):
    """
    [검색 API]
    1. 프롬프트에서 카테고리 키워드 감지 (베이스, 립 등)
    2. 해당 카테고리로 ChromaDB where 필터링 적용
    3. 결과 반환
    """

    # [핵심] 프롬프트 분석하여 카테고리 필터 결정
    target_category = None
    prompt = request.prompt

    # 키워드 매핑 (Java의 Category Enum과 일치해야 함)
    if any(word in prompt for word in ["베이스", "파데", "파운데이션", "쿠션", "컨실러", "비비", "BB"]):
        target_category = "BASE"
    elif any(word in prompt for word in ["립", "틴트", "루주", "글로스", "밤"]):
        target_category = "LIP"
    elif any(word in prompt for word in ["아이", "섀도우", "라이너", "마스카라", "브로우"]):
        target_category = "EYE"
    elif any(word in prompt for word in ["스킨", "토너", "로션", "크림", "에센스", "세럼"]):
        target_category = "SKIN"

    # ChromaDB 쿼리용 where 절 생성
    where_clause = {}
    if target_category:
        where_clause = {"category": target_category}  # 정확히 해당 카테고리만 검색

    try:
        results = collection.query(
            query_texts=[request.prompt],
            n_results=100,  # 필터링을 위해 넉넉히 조회
            where=where_clause  # [핵심] 카테고리 필터링 적용
        )
    except Exception as e:
        print(f"[Error] ChromaDB Search Failed: {e}")
        return {"results": []}

    searched_items = []
    seen_product_ids = set()

    if results["ids"]:
        count = len(results["ids"][0])

        for i in range(count):
            if len(searched_items) >= 10:
                break

            meta = results["metadatas"][0][i]
            p_id = meta.get("product_id")
            user_desc = meta.get("user_description", "")

            # 1. 퍼스널 컬러 필터링 (Python 레벨)
            if request.personal_color not in user_desc:
                continue

            # 2. 중복 상품 제거
            if p_id in seen_product_ids:
                continue

            seen_product_ids.add(p_id)

            doc = results["documents"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0

            searched_items.append({
                "product_id": p_id,
                "similarity_distance": distance
            })

    return {
        "results": searched_items
    }