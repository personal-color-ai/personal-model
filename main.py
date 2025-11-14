import base64
import os
from collections import Counter
from typing import Optional

import requests
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import functions as f
import skin_model as m
from recommend.musinsa import get_all_category_ranking, extract_all_categories, find_tab_outlined_module, \
    get_ranking_config

app = FastAPI()

origins = [
    "http://localhost:3000"  # 스프링 부트 애플리케이션이 실행 중인 도메인
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/image")
async def image(data: dict):

    try:
        image_data = data["image"]
        decoded_image = base64.b64decode(image_data.split(",")[1])

        with open("saved.jpg","wb") as fi:
            fi.write(decoded_image)
      
        f.save_skin_mask("saved.jpg")
   
        ans = m.get_season("temp.jpg")
        os.remove("temp.jpg")
        os.remove("saved.jpg")
   
        if ans == 3:
            ans += 1
        elif ans == 0:
            ans = 3

        test = {'result': ans}
        encoded_data = base64.b64encode(str(test).encode('utf-8')).decode('utf-8')
 
        # response = requests.post('http://localhost:3000/output',json={'encodedData':encoded_data})
        return JSONResponse(content={"message":"complete", 'encodedData':encoded_data,  'result': ans})
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="fail")


@app.post("/lip")
async def lip(data: dict):
    try:
        image_data = data["image"]
        decoded_image = base64.b64decode(image_data.split(",")[1])
       
        with open("saved.jpg","wb") as fi:
            fi.write(decoded_image)
        
        path = r"saved.jpg"
       
        rgb_codes = f.get_rgb_codes(path)  #check point
     
        random_rgb_codes = f.filter_lip_random(rgb_codes,40) #set number of randomly picked sample as 40

        os.remove("saved.jpg")
     
        types = Counter(f.calc_dis(random_rgb_codes))
    
        max_value_key = max(types, key=types.get)
        print(max_value_key)
        if max_value_key == 'sp':
            result = 1
        elif max_value_key == 'su':
            result = 2
        elif max_value_key == 'au':
            result = 3
        elif max_value_key == 'win':
            result = 4
        
        data = {'image':image_data,'result':result}
        encoded_data = base64.b64encode(str(data).encode('utf-8')).decode('utf-8')        
        response = requests.post("http://localhost:3000/output2", json={'encodedData':encoded_data})
        
        print(response)
        
        return JSONResponse(content={"message":"complete"})
    except Exception as e:
        raise HTTPException(status_code=500, detail="fail")


# ----- FastAPI 엔드포인트 -----
BASE_SEARCH_URL = "https://www.musinsa.com/search/goods"

HEADERS = {
    # 정중한 UA + 너무 자주 호출하지 않기 (sleep / rate limit 직접 걸어주세요)
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}




@app.get("/musinsa/categories")
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


@app.get("/musinsa/ranking/all")
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
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("musinsa ranking all error:", repr(e))
        raise HTTPException(status_code=500, detail="랭킹 조회 중 오류가 발생했습니다.")