# Neuromorphic Retention Search Engine

발표용으로 바로 배포할 수 있는 Streamlit 웹사이트입니다. 사용자는 브라우저에서 뉴로모픽 소자의 소재·공정·구동 조건을 입력하고 예상 retention time을 확인할 수 있습니다.

## 웹사이트 기능

1. **조건 검색**: 소재·공정·전해질·구동 조건 입력 → 예상 `Tau_ms` 출력
2. **목표값 검색**: 원하는 retention time 범위 입력 → 후보 조건 Top N 추천
3. **유사 실험 검색**: 기존 문헌 기반 후보 조건 필터링
4. **발표/사업화 포인트**: 완성도, 실현가능성, 사회적 기여도 설명 문구 제공

## 실제 웹사이트로 배포하는 방법

### 1. GitHub에 업로드

1. GitHub에서 새 repository 생성
2. 이 폴더 안의 파일을 그대로 업로드
3. 반드시 아래 파일/폴더가 포함되어야 함

```text
app.py
model_utils.py
requirements.txt
runtime.txt
.streamlit/config.toml
models/retention_model.joblib
models/input_schema.json
models/training_summary.json
models/candidate_predictions.csv
```

### 2. Streamlit Community Cloud에서 배포

1. Streamlit Community Cloud 접속
2. `New app` 클릭
3. GitHub repository 선택
4. Main file path에 `app.py` 입력
5. `Deploy` 클릭

배포가 끝나면 `https://...streamlit.app` 형태의 실제 접속 링크가 생성됩니다. 발표 자료에는 이 링크와 화면 캡처를 넣으면 됩니다.

## 발표에서 쓸 설명

> 기존 결과물이 모델 성능을 확인하는 단계였다면, 이번에는 이를 사용자가 직접 조작할 수 있는 웹 기반 검색 엔진으로 확장했습니다. 사용자는 소재, 공정 조건, 전기적 구동 조건을 입력하고 예상 retention time을 바로 확인할 수 있습니다. 또한 원하는 retention time 범위를 입력하면 기존 문헌 기반 후보 조건을 모델로 재평가해 목표값에 가까운 조합을 추천합니다.

## Q&A 방어 문장

> 현재 모델은 모든 미지의 소재 조합을 완벽하게 예측하는 상용 설계 엔진은 아닙니다. 다만 문헌 기반 데이터 범위 안에서 반복 실험 전에 우선 검토할 조건을 선별하는 연구자용 screening tool입니다.
