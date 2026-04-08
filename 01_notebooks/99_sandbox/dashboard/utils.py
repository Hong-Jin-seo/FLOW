"""
utils.py
--------
app.py 에서 import 해서 쓰는 모든 유틸리티를 모아 둔 파일입니다.
  - DATA     : 정적 데이터 (KPI, 차트용, 유저 목록)
  - Charts   : Plotly Figure 생성 함수
  - UI       : Streamlit 렌더링 컴포넌트 함수
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

KPI_DATA = [
    {
        "icon": "👥", "title": "총 사용자",
        "value": "2,050", "change": "+5.2%",
        "icon_bg": "#eff6ff", "change_color": "#16a34a",
    },
    {
        "icon": "🚪", "title": "예상 이탈 사용자",
        "value": "247", "change": "+12.3%",
        "icon_bg": "#fff1f2", "change_color": "#dc2626",
    },
    {
        "icon": "📊", "title": "구독 유지율",
        "value": "87.9%", "change": "-1.1%",
        "icon_bg": "#faf5ff", "change_color": "#ef4444",
    },
]

MONTHLY_TREND = pd.DataFrame([
    {"month": "1월",  "actual": 52, "predicted": 55},
    {"month": "2월",  "actual": 58, "predicted": 60},
    {"month": "3월",  "actual": 61, "predicted": 63},
    {"month": "4월",  "actual": 66, "predicted": 67},
    {"month": "5월",  "actual": 71, "predicted": 70},
    {"month": "6월",  "actual": 69, "predicted": 71},
    {"month": "7월",  "actual": 74, "predicted": 75},
    {"month": "8월",  "actual": 77, "predicted": 76},
    {"month": "9월",  "actual": 74, "predicted": 73},
    {"month": "10월", "actual": 72, "predicted": 71},
    {"month": "11월", "actual": 70, "predicted": 69},
    {"month": "12월", "actual": 68, "predicted": 67},
])

RISK_SEGMENTS = [
    {"label": "높은 위험", "count": 247, "percent": 12, "color": "#ff5a5f"},
    {"label": "중간 위험", "count": 589, "percent": 29, "color": "#ff8a00"},
    {"label": "낮은 위험", "count": 823, "percent": 40, "color": "#f4c000"},
    {"label": "안전",      "count": 391, "percent": 19, "color": "#22c55e"},
]

GENRES = [
    {"label": "드라마", "value": 38, "color": "#ef4444"},
    {"label": "영화",   "value": 27, "color": "#8b5cf6"},
    {"label": "예능",   "value": 19, "color": "#3b82f6"},
    {"label": "다큐",   "value": 10, "color": "#f59e0b"},
    {"label": "애니",   "value":  6, "color": "#10b981"},
]

OTT_USAGE = [
    {"label": "매일",      "value": 42},
    {"label": "주 3~4회",  "value": 28},
    {"label": "주 1~2회",  "value": 18},
    {"label": "월 1~3회",  "value": 12},
]

CHURN_DRIVERS = [
    {"label": "비활성 일수",         "value": 32},
    {"label": "주간 시청 시간 감소", "value": 24},
    {"label": "로그인 빈도 감소",    "value": 18},
    {"label": "콘텐츠 다양성 부족",  "value": 14},
]

HIGH_RISK_USERS = [
    {
        "name": "Sarah Johnson",   "email": "sarah.j@email.com",
        "risk": 87, "inactive": 18, "lastActiveDate": "2026.02.17",
        "favoriteGenre": "드라마", "subscriptionMonths": 14,
        "recentWatchHours": "1.8시간", "preferredTime": "주말 저녁",
        "churnReason": "2주 이상 미접속, 시청 시간 급감",
        "avatar": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=80&q=80",
    },
    {
        "name": "Michael Chen",    "email": "michael.chen@email.com",
        "risk": 82, "inactive": 15, "lastActiveDate": "2026.02.20",
        "favoriteGenre": "영화",   "subscriptionMonths": 10,
        "recentWatchHours": "2.3시간", "preferredTime": "평일 밤",
        "churnReason": "로그인 빈도 감소, 콘텐츠 다양성 부족",
        "avatar": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=80&q=80",
    },
    {
        "name": "Emma Rodriguez",  "email": "emma.r@email.com",
        "risk": 79, "inactive": 22, "lastActiveDate": "2026.02.13",
        "favoriteGenre": "예능",   "subscriptionMonths": 7,
        "recentWatchHours": "1.1시간", "preferredTime": "주중 오후",
        "churnReason": "장기 비활성, 재방문율 하락",
        "avatar": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?auto=format&fit=crop&w=80&q=80",
    },
    {
        "name": "Lisa Anderson",   "email": "lisa.anderson@email.com",
        "risk": 76, "inactive": 12, "lastActiveDate": "2026.02.23",
        "favoriteGenre": "드라마", "subscriptionMonths": 19,
        "recentWatchHours": "2.9시간", "preferredTime": "평일 저녁",
        "churnReason": "주간 시청 시간 감소",
        "avatar": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=80&q=80",
    },
    {
        "name": "James Wilson",    "email": "james.w@email.com",
        "risk": 73, "inactive": 16, "lastActiveDate": "2026.02.19",
        "favoriteGenre": "다큐",   "subscriptionMonths": 5,
        "recentWatchHours": "1.5시간", "preferredTime": "새벽",
        "churnReason": "최근 시청 이력 감소, 로그인 빈도 하락",
        "avatar": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?auto=format&fit=crop&w=80&q=80",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS  (Plotly Figure 반환 함수)
# ══════════════════════════════════════════════════════════════════════════════

def make_trend_chart(df: pd.DataFrame) -> go.Figure:
    """월별 이탈 추세 꺾은선 차트."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["month"], y=df["predicted"],
        mode="lines+markers", name="예측",
        line=dict(color="#a855f7", width=3),
        marker=dict(size=7, color="#a855f7"),
    ))
    fig.add_trace(go.Scatter(
        x=df["month"], y=df["actual"],
        mode="lines+markers", name="실제",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=8, color="#3b82f6"),
    ))
    fig.update_layout(
        height=290,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", range=[45, 82]),
    )
    return fig


def make_risk_donut(segments: list[dict]) -> go.Figure:
    """위험 세분화 도넛 차트."""
    total = sum(s["count"] for s in segments)
    fig = go.Figure(go.Pie(
        labels=[s["label"] for s in segments],
        values=[s["count"] for s in segments],
        hole=0.62,
        marker=dict(colors=[s["color"] for s in segments], line=dict(width=0)),
        textinfo="none",
        hovertemplate="%{label}: %{value}명 (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[dict(
            text=f"<b>{total:,}</b><br>총 사용자",
            x=0.5, y=0.5, font_size=13, showarrow=False, font_color="#111",
        )],
    )
    return fig


def make_genre_donut(genres: list[dict]) -> go.Figure:
    """콘텐츠 장르 비중 도넛 차트."""
    fig = go.Figure(go.Pie(
        labels=[g["label"] for g in genres],
        values=[g["value"] for g in genres],
        hole=0.55,
        marker=dict(colors=[g["color"] for g in genres], line=dict(width=0)),
        textinfo="none",
        hovertemplate="%{label}: %{value}%<extra></extra>",
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI  (Streamlit 렌더링 컴포넌트)
# ══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    """전역 CSS를 주입합니다. app.py 최상단에서 한 번만 호출하세요."""
    st.markdown("""
    <style>
    /* 전체 배경 */
    .stApp { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); }

    /* Streamlit 기본 상단 여백 제거 → 헤더가 꽉 차게 */
    .stApp > header { display: none; }
    #root > div:first-child { padding-top: 0 !important; }
    .stMainBlockContainer {
        padding-top: 0 !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* ── st.container(border=True) 카드 스타일 오버라이드 ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white !important;
        border-radius: 20px !important;
        border: 1px solid #e9ecef !important;
        box-shadow: 0 2px 14px rgba(0,0,0,0.06) !important;
        padding: 8px !important;
    }
    /* 카드 내부 여백 */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 8px 6px !important;
    }

    /* 카드 타이틀 (markdown으로 주입) */
    .card-title { font-size: 1rem; font-weight: 700; color: #111;
                  margin: 0 0 2px 0; padding: 0; }
    .card-sub   { font-size: 0.8rem; color: #6b7280;
                  margin: 0 0 10px 0; padding: 0; }

    /* 진행 바 */
    .bar-row   { margin-bottom: 13px; }
    .bar-label { display: flex; justify-content: space-between;
                 font-size: 0.83rem; margin-bottom: 5px; }
    .bar-track { background: #f3f4f6; border-radius: 8px;
                 height: 10px; overflow: hidden; }

    /* 위험 세분화 범례 */
    .seg-row   { display: flex; justify-content: space-between; align-items: center;
                 font-size: 0.82rem; margin-bottom: 6px; }

    /* 경고 박스 */
    .warn-box  { background: #fffbeb; border: 1px solid #fde68a;
                 border-radius: 14px; padding: 12px 16px;
                 margin-top: 14px; font-size: 0.84rem; color: #374151; }

    /* 이탈 요인 박스 */
    .churn-box { background: #fff0f0; border: 1px solid #fecaca;
                 border-radius: 12px; padding: 12px 16px;
                 margin-top: 10px; font-size: 0.84rem; color: #374151; }

    /* Plotly 차트 여백 제거 */
    div[data-testid="stVerticalBlockBorderWrapper"] .stPlotlyChart {
        margin-top: -8px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown("""
    <div style="background:#000;color:white;padding:18px 28px;
                border-radius:0 0 24px 24px;display:flex;align-items:center;
                gap:16px;margin-bottom:20px;box-shadow:0 4px 24px rgba(0,0,0,0.22);">
        <div style="width:34px;height:34px;background:#E50914;
                    border-radius:7px;flex-shrink:0;display:flex;
                    align-items:center;justify-content:center;
                    font-weight:900;font-size:0.85rem;color:white;letter-spacing:-0.5px;">N</div>
        <div style="width:1px;height:28px;background:#444"></div>
        <span style="font-size:1.1rem;font-weight:700;letter-spacing:-0.3px;">
            Netflix 이탈 예측 분석 대시보드
        </span>
        <span style="margin-left:auto;font-size:0.78rem;color:#888;">
            최종 업데이트: 2026년 3월 6일 · 오전 9:42
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_toolbar() -> None:
    """사이드바 대신 헤더 바로 아래 인라인 툴바로 업로드·필터를 배치합니다."""
    with st.container():
        col_upload, col_risk, col_genre, col_date = st.columns([2, 2, 2, 1.5])

        with col_upload:
            uploaded = st.file_uploader(
                "📁 데이터 업로드",
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
                label_visibility="visible",
            )
            if uploaded:
                st.caption(f"✅ {len(uploaded)}개 파일 업로드됨: " +
                           ", ".join(f.name for f in uploaded))

        with col_risk:
            st.multiselect(
                "🔴 위험 등급 필터",
                ["높은 위험", "중간 위험", "낮은 위험", "안전"],
                default=["높은 위험", "중간 위험", "낮은 위험", "안전"],
            )

        with col_genre:
            st.multiselect(
                "🎬 장르 필터",
                ["드라마", "영화", "예능", "다큐", "애니"],
                default=["드라마", "영화", "예능", "다큐", "애니"],
            )

        with col_date:
            st.selectbox(
                "📅 기간",
                ["최근 1개월", "최근 3개월", "최근 6개월", "전체"],
                index=1,
            )

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)


def render_kpi_card(col, item: dict) -> None:
    with col:
        with st.container(border=True):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        align-items:flex-start;margin-bottom:18px">
                <div style="width:48px;height:48px;border-radius:14px;
                            background:{item['icon_bg']};display:flex;
                            align-items:center;justify-content:center;
                            font-size:1.4rem">{item['icon']}</div>
                <span style="font-size:0.85rem;font-weight:700;
                             color:{item['change_color']}">{item['change']}</span>
            </div>
            <p style="font-size:0.8rem;color:#6b7280;margin:0 0 6px">{item['title']}</p>
            <p style="font-size:2rem;font-weight:800;color:#111;
                      margin:0;letter-spacing:-1px">{item['value']}</p>
            """, unsafe_allow_html=True)


def render_risk_donut(segments: list[dict]) -> None:
    with st.container(border=True):
        st.markdown('<p class="card-title">🎯 이탈 위험 세분화</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">위험도별 사용자 분포</p>', unsafe_allow_html=True)
        st.plotly_chart(make_risk_donut(segments), use_container_width=True)
        for seg in segments:
            st.markdown(f"""
            <div class="seg-row">
                <div style="display:flex;align-items:center;gap:8px">
                    <span style="width:11px;height:11px;border-radius:50%;
                                 background:{seg['color']};display:inline-block"></span>
                    <span style="color:#374151">{seg['label']}</span>
                </div>
                <div style="display:flex;gap:14px;color:#6b7280">
                    <b style="color:#111">{seg['count']}</b>
                    <span>{seg['percent']}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def _bar_html(label: str, value: int, color: str) -> str:
    """진행 바 HTML 조각을 반환합니다 (내부 헬퍼)."""
    return f"""
    <div class="bar-row">
        <div class="bar-label">
            <span style="color:#374151">{label}</span>
            <b style="color:#111">{value}%</b>
        </div>
        <div class="bar-track">
            <div style="height:100%;width:{value}%;background:{color};
                        border-radius:8px"></div>
        </div>
    </div>"""


def render_ott_usage(data: list[dict]) -> None:
    with st.container(border=True):
        st.markdown('<p class="card-title">▶️ OTT 이용 빈도</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">최근 3개월 기준</p>', unsafe_allow_html=True)
        for item in data:
            st.markdown(_bar_html(item["label"], item["value"], "#3b82f6"),
                        unsafe_allow_html=True)


def render_genre_chart(genres: list[dict]) -> None:
    with st.container(border=True):
        st.markdown('<p class="card-title">🎬 콘텐츠 장르</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">주요 시청 장르 비중</p>', unsafe_allow_html=True)
        st.plotly_chart(make_genre_donut(genres), use_container_width=True)
        for g in genres:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        font-size:0.82rem;margin-bottom:5px">
                <div style="display:flex;align-items:center;gap:7px">
                    <span style="width:10px;height:10px;border-radius:50%;
                                 background:{g['color']};display:inline-block"></span>
                    <span style="color:#374151">{g['label']}</span>
                </div>
                <b style="color:#111">{g['value']}%</b>
            </div>
            """, unsafe_allow_html=True)


def render_churn_drivers(drivers: list[dict]) -> None:
    with st.container(border=True):
        st.markdown('<p class="card-title">📉 이탈 예측 주요 요인</p>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">이탈 가능성에 가장 큰 영향을 주는 행동 지표</p>',
                    unsafe_allow_html=True)
        for item in drivers:
            st.markdown(_bar_html(item["label"], item["value"], "#ef4444"),
                        unsafe_allow_html=True)
        st.markdown(
            '<div class="warn-box">'
            '⚠️ 최근 2주 이상 비활성 고객군의 이탈 위험이 가장 높습니다.'
            '</div>',
            unsafe_allow_html=True,
        )


def render_high_risk_users(users: list[dict]) -> None:
    with st.container(border=True):
        st.markdown('<p class="card-title">🚨 높은 이탈 위험 사용자</p>', unsafe_allow_html=True)

        for i, user in enumerate(users):
            risk = user["risk"]
            dot  = "🔴" if risk >= 85 else "🟠" if risk >= 78 else "🟡"
            label = (
                f"{dot} {user['name']}  ·  비활성 {user['inactive']}일  ·  "
                f"{user['favoriteGenre']}  |  위험도 {risk}%"
            )
            with st.expander(label, expanded=(i == 0)):
                img_col, info_col = st.columns([1, 6])

                with img_col:
                    st.image(user["avatar"], width=72)

                with info_col:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**📧 이메일**<br>{user['email']}",
                                unsafe_allow_html=True)
                    c2.markdown(f"**📅 마지막 활동일**<br>{user['lastActiveDate']}",
                                unsafe_allow_html=True)
                    c3.markdown(f"**⏰ 비활성 일수**<br>{user['inactive']}일",
                                unsafe_allow_html=True)
                    c4.markdown(f"**🎬 주요 장르**<br>{user['favoriteGenre']}",
                                unsafe_allow_html=True)

                    d1, d2, d3, _ = st.columns(4)
                    d1.markdown(f"**📆 구독 개월 수**<br>{user['subscriptionMonths']}개월",
                                unsafe_allow_html=True)
                    d2.markdown(f"**🕐 최근 30일 시청**<br>{user['recentWatchHours']}",
                                unsafe_allow_html=True)
                    d3.markdown(f"**🌙 선호 시간대**<br>{user['preferredTime']}",
                                unsafe_allow_html=True)

                st.markdown(
                    f'<div class="churn-box">'
                    f'<b>⚠️ 주요 이탈 요인:</b> {user["churnReason"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
