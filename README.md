# 📑 입찰메이트

<div align="center">
  <img src=".images/baner_image.png" width="100%">
</div>



Repo URL : [![Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/codeit-ai12-project2-team4/rfp-rag-system)

팀 보고서 URL : `[노션 팀 보고서 링크]`

협업일지 : `[일지 PDF 링크]`

<br/>

## 📋 Project Overview

### AI 입찰메이트 RAG 시스템

RFP 문서의 복잡한 내용과 메타데이터를 효과적으로 추출하고 요약하여, 기업 및 정부 입찰에 최적화된 맞춤형 기회를 신속하게 제공하는 사내 AI 컨설턴트 구축 프로젝트입니다.

---

**입찰메이트**는 공공입찰 컨설팅 서비스를 제공하는 B2G 전문 스타트업입니다. 나라장터를 비롯한 여러 채널에는 하루에도 수백 건의 RFP(제안요청서)가 새로 올라오고, 그중 고객사에 맞는 입찰 기회를 찾아 추천하는 것이 저희 컨설팅 업무의 시작입니다. 문제는 RFP 한 건이 짧게는 수십 페이지, 길게는 수백 페이지에 달하고, 참가자격·제출서류·평가배점·과업기간 같은 핵심 정보가 문서마다 다른 서식과 표 안에 흩어져 있다는 점입니다. 컨설턴트가 이를 일일이 읽어 파악하다 보면 시간이 오래 걸릴 뿐 아니라 정보를 놓치거나 잘못 읽는 실수도 생기기 쉽습니다.

이 프로젝트는 이런 과정을 AI가 대신하도록 만드는 사내 RAG(Retrieval-Augmented Generation) Q&A 시스템입니다. 원본 RFP 문서(hwp/pdf)를 정제하고 청크 단위로 잘라 벡터 인덱스로 구축한 뒤, 질문이 들어오면 관련 근거를 검색하고 이를 바탕으로 LLM이 답변을 생성하는 검색-생성 파이프라인으로 동작합니다. 컨설턴트는 더 이상 방대한 원문을 처음부터 끝까지 읽지 않아도, 필요한 정보를 근거 문구와 함께 바로 확인할 수 있습니다. 결과적으로 입찰 준비 시간을 크게 줄이고 누락·오독 위험을 낮춰, 더 많은 고객사에게 더 빠르게 입찰 기회를 제안할 수 있게 됩니다.

<br/>

## 😄 Team Member

<div align="center">

<table>
    <tr align="center">
        <td><img src=".images/hs_image.png" width="120" height="120" style="border-radius:50%;"></td>
        <td><img src=".images/ch_image.png" width="120" height="120" style="border-radius:50%;"></td>
        <td><img src="https://via.placeholder.com/120?text=%20" width="120" height="120" style="border-radius:50%;"></td>
        <td><img src=".images/ms_image.png" width="120" height="120" style="border-radius:50%;"></td>
        <td><img src=".images/jy_image.png" width="120" height="120" style="border-radius:50%;"></td>
    </tr>
    <tr align="center">
        <td><b>권혁상</b></td>
        <td><b>안찬형</b></td>
        <td><b>이하람</b></td>
        <td><b>최민식</b></td>
        <td><b>최중열</b></td>
    </tr>
    <tr align="center">
        <td>
            <img src="https://img.shields.io/badge/Data%20Engineer-E67E22?style=flat-square">
        </td>
        <td>
            <img src="https://img.shields.io/badge/Project%20Manager-2E8B57?style=flat-square">
        </td>
        <td>
            <!-- 이하람  -->
        </td>
        <td>
            <img src="https://img.shields.io/badge/Generation%20Engineer-1E6FD9?style=flat-square">
        </td>
        <td>
            <img src="https://img.shields.io/badge/Retrieval%20Engineer-9B59B6?style=flat-square">
        </td>
    </tr>
</table>

</div>


<br/>

## 🔗 Reference

| 문서 | 링크 |
| :--: | :--: |
| 📎 EDA README | # EDA README 경로 |
| 📎 src 통합 README | [바로가기](src/README.md) |
| 📎 pipeline README | # pipeline README 경로 |

<br/>

## 🗂️ Project Structure

```
📦 rfp-rag-system
┣ 📂 config
┃ ┣ 📝 __init__.py
┃ ┣ 📝 model_config.py
┃ ┗ 📝 settings.py
┣ 📂 data
┃ ┗ 📃 eval_qa.json
┣ 📂 eda
┣ 📂 outputs            (gitignore)
┣ 📂 src
┃ ┣ 📂 models
┃ ┃ ┣ 📝 __init__.py
┃ ┃ ┣ 📝 embed.py
┃ ┃ ┣ 📝 rerank.py
┃ ┃ ┣ 📝 llm.py
┃ ┃ ┗ 📝 health.py
┃ ┣ 📂 pieces
┃ ┃ ┣ 📝 __init__.py
┃ ┃ ┣ 📝 base.py
┃ ┃ ┣ 📝 search.py
┃ ┃ ┗ 📝 refine.py
┃ ┣ 📂 evaluation
┃ ┃ ┣ 📝 __init__.py
┃ ┃ ┣ 📝 evalset.py
┃ ┃ ┣ 📝 retrieval.py
┃ ┃ ┗ 📝 generation.py
┃ ┣ 📂 preprocessing
┃ ┃ ┣ 📝 __init__.py
┃ ┃ ┣ 📝 clean.py
┃ ┃ ┣ 📝 hwp.py
┃ ┃ ┣ 📝 hwp_table.py
┃ ┃ ┣ 📝 pdf.py
┃ ┃ ┣ 📝 run.py
┃ ┃ ┗ 📝 toc.py
┃ ┣ 📝 __init__.py
┃ ┣ 📝 chunking.py
┃ ┣ 📝 vectorstore.py
┃ ┣ 📝 resources.py
┃ ┣ 📝 retriever.py
┃ ┣ 📝 generation.py
┃ ┣ 📝 pipeline.py
┃ ┗ 📃 README.md
┣ 🔧 .env.example
┣ 🔧 .gitignore
┣ 📝 main.py
┣ 📃 PIPELINE.md
┣ 📃 README.md
┗ 📃 requirements.txt
```

<br/>

## 🌿 Team Git Rule's

### ✅ Code Style

`Black` or `Ruff` 확장 프로그램 사용

### ✍️ Comments

- Google Style
- 간단한 method의 경우 `$DESCRIPTION$`만 사용

```python
"""
$DESCRIPTION$

Args:
    $PARAMS$: param

Returns:
    $RETURN$:

Raises:
    $EXCEPTION$
"""
```

### 📝 Commit Rule

| Prefix | 설명 |
| :--: | :-- |
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 (README 등) |
| `style` | 포맷 변경 (코드 동작 무관) |
| `refactor` | 리팩토링 |
| `test` | 테스트 추가/수정 |
| `chore` | 빌드, 패키지 설정 변경 |