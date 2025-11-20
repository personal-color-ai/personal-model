import fastapi
import functions as fn
import cv2
from PIL import Image
from collections import Counter
import numpy as np
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Dict
import base64
import skin_model as m
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image as PILImage
            

app = FastAPI(
    title="Personal Color Analysis API",
    description="퍼스널컬러 분석 API - 피부색과 립컬러를 분석하여 시즌 타입을 분류합니다.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

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


# 응답 모델 정의
class SeasonProbs(BaseModel):
    """시즌별 확률"""
    spring: float
    summer: float
    autumn: float
    winter: float


class AnalysisResult(BaseModel):
    """분석 결과"""
    result: str  # "spring", "summer", "autumn", "winter" 중 하나
    probs: SeasonProbs


class AnalyzeResponse(BaseModel):
    """통합 분석 API 응답"""
    message: str
    image: AnalysisResult
    lip: AnalysisResult
    eye: AnalysisResult


class ImageResponse(BaseModel):
    """이미지 분석 API 응답"""
    message: str
    result: int
    probs: list
    chart: str


class LipResponse(BaseModel):
    """립 분석 API 응답"""
    message: str
    result: int
    probs: list
    chart: str


def number_to_season(num: int) -> str:
    """숫자를 시즌 문자열로 변환"""
    season_map = {
        1: "spring",
        2: "summer",
        3: "autumn",
        4: "winter"
    }
    return season_map.get(num, "unknown")


def reorder_probs_to_season_order(probs: list, model_order: list = [3, 1, 2, 4]) -> dict:
    """
    모델의 확률 배열을 시즌 순서(spring, summer, autumn, winter)로 재정렬
    
    Args:
        probs: 모델에서 반환된 확률 배열 (길이 4)
        model_order: 모델 인덱스 [0,1,2,3]이 의미하는 시즌 번호 [3,1,2,4]
                    (autumn, spring, summer, winter)
    
    Returns:
        {"spring": float, "summer": float, "autumn": float, "winter": float}
    """
    # 모델 인덱스 0,1,2,3 → 시즌 번호 3,1,2,4 → autumn, spring, summer, winter
    season_probs = {
        "spring": probs[1],   # 모델 인덱스 1 → spring
        "summer": probs[2],   # 모델 인덱스 2 → summer
        "autumn": probs[0],   # 모델 인덱스 0 → autumn
        "winter": probs[3]    # 모델 인덱스 3 → winter
    }
    return season_probs


@app.post(
    "/image",
    response_model=ImageResponse,
    summary="피부색 분석",
    description="이미지에서 피부 영역을 추출하여 퍼스널컬러 시즌 타입을 분석합니다.",
    tags=["분석"]
)
async def image(file: UploadFile = File(..., description="분석할 이미지 파일 (JPG, PNG 등)")):
    try:
        # 1️⃣ 파일 저장
        contents = await file.read()
        save_path = "saved.jpg"
        with open(save_path, "wb") as out:  # ✅ out으로 변경
            out.write(contents)

        # 2️⃣ 피부 마스크 생성 (temp.jpg 생성 가정)
        fn.save_skin_mask(save_path)

        # 3️⃣ 퍼스널컬러 분류 (클래스 및 확률)
        ans = m.get_season("temp.jpg")
        probs = m.get_season_probs("temp.jpg")  # length 4, order 0..3

        # 4️⃣ 결과 후처리
        if ans == 3:
            ans += 1
        elif ans == 0:
            ans = 3

        # 5️⃣ 확률 막대그래프 생성(Base64)
        labels = ["0", "1", "2", "3"]
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax.bar(labels, probs, color=["#ffb74d", "#64b5f6", "#a1887f", "#90caf9"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Season Probabilities")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
        chart_data_url = f"data:image/png;base64,{chart_b64}"

        return {
            "message": "complete",
            "result": ans,
            "probs": probs,
            "chart": chart_data_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 5️⃣ 임시 파일 삭제
        for path in ("saved.jpg", "temp.jpg"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


# @app.post("/image")
# async def image(data: dict):

#     try:
#         image_data = data["image"]
#         decoded_image = base64.b64decode(image_data.split(",")[1])

#         with open("saved.jpg","wb") as fi:
#             fi.write(decoded_image)
      
#         f.save_skin_mask("saved.jpg")
   
#         ans = m.get_season("temp.jpg")
#         os.remove("temp.jpg")
#         os.remove("saved.jpg")
   
#         if ans == 3:
#             ans += 1
#         elif ans == 0:
#             ans = 3

#         test = {'result': ans}
#         encoded_data = base64.b64encode(str(test).encode('utf-8')).decode('utf-8')
 
#         # response = requests.post('http://localhost:3000/output',json={'encodedData':encoded_data})
#         return JSONResponse(content={"message":"complete", 'encodedData':encoded_data,  'result': ans})
        
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail="fail")


@app.post(
    "/lip",
    response_model=LipResponse,
    summary="립컬러 분석",
    description="이미지에서 립 영역을 추출하여 퍼스널컬러 시즌 타입을 분석합니다.",
    tags=["분석"]
)
async def lip(file: UploadFile = File(..., description="분석할 이미지 파일 (JPG, PNG 등)")):
    try:
        # 1️⃣ 파일 저장
        contents = await file.read()
        save_path = "saved.jpg"
        with open(save_path, "wb") as out:
            out.write(contents)
        
        # 2️⃣ RGB 코드 추출 및 분석
        rgb_codes = fn.get_rgb_codes(save_path)
        random_rgb_codes = fn.filter_lip_random(rgb_codes, 40)  # 40개 샘플 랜덤 선택
        
        # 3️⃣ 각 샘플의 타입 계산
        types = Counter(fn.calc_dis(random_rgb_codes))
        total_samples = sum(types.values())
        
        # 4️⃣ 퍼센트 계산 (sp, su, au, win 순서)
        probs = [
            types.get('sp', 0) / total_samples,  # Spring (result=1)
            types.get('su', 0) / total_samples,  # Summer (result=2)
            types.get('au', 0) / total_samples,  # Autumn (result=3)
            types.get('win', 0) / total_samples  # Winter (result=4)
        ]
        
        # 5️⃣ 가장 높은 확률을 가진 타입 결정
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
        
        # 6️⃣ 확률 막대그래프 생성(Base64)
        labels = ["1", "2", "3", "4"]  # Spring, Summer, Autumn, Winter
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        ax.bar(labels, probs, color=["#ffb74d", "#64b5f6", "#a1887f", "#90caf9"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Lip Color Probabilities")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
        chart_data_url = f"data:image/png;base64,{chart_b64}"

        return {
            "message": "complete",
            "result": result,
            "probs": probs,
            "chart": chart_data_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 7️⃣ 임시 파일 삭제
        if os.path.exists("saved.jpg"):
            try:
                os.remove("saved.jpg")
            except:
                pass


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="통합 분석 (피부색 + 립컬러 + 눈동자)",
    description="이미지에서 피부색, 립컬러, 눈동자를 동시에 분석하여 각각의 시즌 타입과 확률을 반환합니다. 차트는 포함하지 않으며, 프론트엔드에서 probs 값으로 직접 그려야 합니다.",
    tags=["통합 분석"]
)
async def analyze(file: UploadFile = File(..., description="분석할 이미지 파일 (JPG, PNG 등)")):
    """
    피부색, 립컬러, 눈동자를 한번에 분석하여 결과값과 확률을 반환합니다.
    
    - **image**: 피부색 분석 결과
        - result: "spring", "summer", "autumn", "winter" 중 하나
        - probs: 각 시즌별 확률 (0~1)
    
    - **lip**: 립컬러 분석 결과
        - result: "spring", "summer", "autumn", "winter" 중 하나
        - probs: 각 시즌별 확률 (0~1)
    
    - **eye**: 눈동자 색상 분석 결과
        - result: "spring", "summer", "autumn", "winter" 중 하나
        - probs: 각 시즌별 확률 (0~1)
    
    차트는 포함하지 않습니다.
    """
    try:
        # 1️⃣ 파일 저장
        contents = await file.read()
        save_path = "saved.jpg"
        with open(save_path, "wb") as out:
            out.write(contents)

        # 2️⃣ 피부색 분석 (image)
        fn.save_skin_mask(save_path)
        image_probs_raw = m.get_season_probs("temp.jpg")  # length 4, order 0..3
        
        # 확률을 시즌 순서로 재정렬
        image_probs = reorder_probs_to_season_order(image_probs_raw)
        
        # 재정렬된 확률에서 가장 높은 값을 가진 시즌 찾기
        image_result = max(image_probs, key=image_probs.get)

        # 3️⃣ 립컬러 분석 (lip)
        rgb_codes = fn.get_rgb_codes(save_path)
        random_rgb_codes = fn.filter_lip_random(rgb_codes, 40)  # 40개 샘플 랜덤 선택
        
        types = Counter(fn.calc_dis(random_rgb_codes))
        total_samples = sum(types.values())
        
        # 퍼센트 계산 (spring, summer, autumn, winter 순서)
        lip_probs = {
            "spring": types.get('sp', 0) / total_samples,
            "summer": types.get('su', 0) / total_samples,
            "autumn": types.get('au', 0) / total_samples,
            "winter": types.get('win', 0) / total_samples
        }
        
        # 확률에서 가장 높은 값을 가진 시즌 찾기
        lip_result = max(lip_probs, key=lip_probs.get)

        # 4️⃣ 눈 분석 (eye)
        eye_rgb_codes = fn.get_eye_rgb_codes(save_path)
        
        # 눈 샘플 수가 충분한지 확인하고 랜덤 샘플링
        if len(eye_rgb_codes) > 40:
            random_indices = np.random.randint(0, len(eye_rgb_codes), 40)
            random_eye_rgb_codes = eye_rgb_codes[random_indices]
        else:
            random_eye_rgb_codes = eye_rgb_codes  # 샘플이 적으면 전부 사용
        
        eye_types = Counter(fn.calc_dis(random_eye_rgb_codes))
        eye_total_samples = sum(eye_types.values())
        
        # 퍼센트 계산 (spring, summer, autumn, winter 순서)
        eye_probs = {
            "spring": eye_types.get('sp', 0) / eye_total_samples if eye_total_samples > 0 else 0.25,
            "summer": eye_types.get('su', 0) / eye_total_samples if eye_total_samples > 0 else 0.25,
            "autumn": eye_types.get('au', 0) / eye_total_samples if eye_total_samples > 0 else 0.25,
            "winter": eye_types.get('win', 0) / eye_total_samples if eye_total_samples > 0 else 0.25
        }
        
        # 확률에서 가장 높은 값을 가진 시즌 찾기
        eye_result = max(eye_probs, key=eye_probs.get)

        return {
            "message": "complete",
            "image": {
                "result": image_result,
                "probs": image_probs
            },
            "lip": {
                "result": lip_result,
                "probs": lip_probs
            },
            "eye": {
                "result": eye_result,
                "probs": eye_probs
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 4️⃣ 임시 파일 삭제
        for path in ("saved.jpg", "temp.jpg"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


@app.post(
    "/visualize",
    summary="마스크 시각화",
    description="이미지에서 얼굴 파싱 결과를 시각화합니다. 눈(파란색), 입술(빨간색), 피부(노란색) 영역이 반투명하게 표시됩니다.",
    tags=["시각화"]
)
async def visualize(file: UploadFile = File(..., description="시각화할 이미지 파일 (JPG, PNG 등)")):
    """
    얼굴 파싱 마스크를 시각화하여 반환합니다.
    
    - **눈 영역**: 파란색으로 표시
    - **입술 영역**: 빨간색으로 표시  
    - **피부 영역**: 노란색으로 표시
    
    원본 이미지 위에 각 영역이 반투명하게 오버레이됩니다.
    """
    try:
        # 1️⃣ 파일 저장
        contents = await file.read()
        save_path = "saved_vis.jpg"
        with open(save_path, "wb") as out:
            out.write(contents)

        # 2️⃣ 마스크 시각화
        vis_img = fn.visualize_masks(save_path, "mask_vis_temp.jpg")
        
        # 3️⃣ Base64로 인코딩
        vis_img_pil = PILImage.fromarray(vis_img)
        buf = BytesIO()
        vis_img_pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_b64}"

        return {
            "message": "complete",
            "image": img_data_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 4️⃣ 임시 파일 삭제
        for path in ("saved_vis.jpg", "mask_vis_temp.jpg"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

