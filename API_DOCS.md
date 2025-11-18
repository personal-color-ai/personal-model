# Personal Color Analysis API 명세서

## API 접근 방법

FastAPI는 자동으로 API 명세서를 생성합니다. 서버 실행 후 아래 URL로 접근할 수 있습니다:

### 1. Swagger UI (인터랙티브 문서)
```
http://localhost:8000/docs
```
- API를 브라우저에서 직접 테스트할 수 있습니다
- 각 엔드포인트를 클릭하여 파라미터 입력 후 "Try it out" 버튼으로 테스트 가능

### 2. ReDoc (읽기 쉬운 문서)
```
http://localhost:8000/redoc
```
- 더 읽기 쉬운 형태의 문서
- 검색 기능 제공

### 3. OpenAPI JSON 스키마
```
http://localhost:8000/openapi.json
```
- JSON 형식의 OpenAPI 스펙
- 다른 도구로 가져가거나 공유할 때 사용

## API 엔드포인트

### 1. `/analyze` (POST) - 통합 분석 ⭐ 추천
**피부색과 립컬러를 동시에 분석합니다.**

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: 이미지 파일 (JPG, PNG 등)

**Response:**
```json
{
  "message": "complete",
  "image": {
    "result": "summer",
    "probs": {
      "spring": 0.075,
      "summer": 0.632,
      "autumn": 0.250,
      "winter": 0.043
    }
  },
  "lip": {
    "result": "autumn",
    "probs": {
      "spring": 0.025,
      "summer": 0.15,
      "autumn": 0.725,
      "winter": 0.1
    }
  }
}
```

**시즌 타입:**
- `"spring"`: 봄 웜톤
- `"summer"`: 여름 쿨톤
- `"autumn"`: 가을 웜톤
- `"winter"`: 겨울 쿨톤

### 2. `/image` (POST) - 피부색 분석
**피부 영역만 분석합니다.**

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: 이미지 파일

**Response:**
```json
{
  "message": "complete",
  "result": 2,
  "probs": [0.075, 0.632, 0.250, 0.043],
  "chart": "data:image/png;base64,..."
}
```

### 3. `/lip` (POST) - 립컬러 분석
**립 영역만 분석합니다.**

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `file`: 이미지 파일

**Response:**
```json
{
  "message": "complete",
  "result": 3,
  "probs": [0.025, 0.15, 0.725, 0.1],
  "chart": "data:image/png;base64,..."
}
```

## 백엔드에서 API 호출하는 방법

### Python (requests)
```python
import requests

# 파일 경로에서
with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/analyze", files=files)
    result = response.json()
```

### JavaScript/TypeScript (fetch)
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    body: formData
});

const result = await response.json();
```

### Spring Boot (Java)
```java
RestTemplate restTemplate = new RestTemplate();
HttpHeaders headers = new HttpHeaders();
headers.setContentType(MediaType.MULTIPART_FORM_DATA);

MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("file", new ByteArrayResource(imageBytes) {
    @Override
    public String getFilename() {
        return "image.jpg";
    }
});

HttpEntity<MultiValueMap<String, Object>> requestEntity = 
    new HttpEntity<>(body, headers);

ResponseEntity<Map> response = restTemplate.exchange(
    "http://localhost:8000/analyze",
    HttpMethod.POST,
    requestEntity,
    Map.class
);
```

## 주의사항

1. **Content-Type**: 반드시 `multipart/form-data` 형식으로 전송해야 합니다
2. **파일 필드명**: `file`로 고정되어 있습니다
3. **지원 형식**: JPG, PNG 등 이미지 파일 형식
4. **응답 시간**: 이미지 처리로 인해 몇 초 소요될 수 있습니다

## 에러 처리

- **500 Internal Server Error**: 이미지 처리 중 오류 발생
- 응답의 `detail` 필드에 에러 메시지가 포함됩니다

