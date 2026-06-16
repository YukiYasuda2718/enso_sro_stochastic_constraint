from logging import getLogger
from typing import Literal

import numpy as np
import pandas as pd
import scipy.signal

logger = getLogger()


# --- ユーティリティ: day-of-year 1..360 の計算 ---
def _fractional_day_of_year(
    years_1d: np.ndarray, days_per_year: int = 360
) -> np.ndarray:
    """
    years_1d: shape (N,) の年単位の連続時刻（例: 100.5 = 100年と180日目）
    戻り値: 1..days_per_year の整数 day-of-year 配列
    """
    frac = years_1d - np.floor(years_1d)
    day = (frac * days_per_year).astype(int) + 1
    # day==days_per_year+1 の丸め込溢れ対策（理論上ほぼ出ないが念のため）
    day = np.where(day > days_per_year, days_per_year, day)
    return day


# --- コア: イベント抽出（しきい値超え連続 >= min_len） ---
def _find_runs_bool(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    True/False の一次元配列中で True の連続区間 [start, end] を列挙する。
    戻り値: List of (start_index, end_index)  (end は含む)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    # True になり始める地点
    starts = np.where(diff == 1)[0] + 1
    # True が終わる地点
    ends = np.where(diff == -1)[0]
    # 先頭/末尾が True で始まる/終わる場合の補正
    if mask[0]:
        logger.debug("Mask starts with True")
        starts = np.r_[0, starts]
    if mask[-1]:
        logger.debug("Mask ends with True")
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts, ends))


def _plateau_centers_local_maxima(y: np.ndarray) -> list[int]:
    """
    連続区間 y (numpy 1D) 内の「局所極大の台地（ε=0）」の中央インデックスを列挙する。
    ここでの局所極大は，台地の両側近傍が（存在すれば）厳密に小さいことを要件とする。
    戻り値: 台地中央のインデックス（y 内のローカル添字）。
    """
    N = len(y)
    if N == 0:
        return []
    peaks = []
    i = 0
    while i < N:
        j = i
        # 台地（同値連）の右端まで伸ばす
        while j + 1 < N and y[j + 1] == y[i]:
            j += 1
        left_ok = (i == 0) or (y[i - 1] < y[i])
        right_ok = (j == N - 1) or (y[j + 1] < y[j])
        if left_ok and right_ok:
            center = int(np.round((i + j) / 2.0))
            peaks.append(center)
        i = j + 1
    return peaks


def _plateau_center_global_max(y: np.ndarray) -> int | None:
    """
    区間 y 内の「グローバル最大値の台地」の中央インデックス（ローカル添字）を 1 つ返す。
    """
    if len(y) == 0:
        return None
    vmax = np.max(y)
    eq = np.where(y == vmax)[0]
    if eq.size == 0:
        return None
    i = eq[0]
    j = eq[-1]
    center = int(np.round((i + j) / 2.0))
    return center


def _detect_peak_counts_for_debugging(
    years: np.ndarray,
    smoothed_series: pd.Series,
    std_value: float,
    n_per_month: int = 30,
    mode: Literal["elnino", "lanina"] = "elnino",
    peak_mode: Literal["max_only", "all_peaks"] = "max_only",
    debug: bool = False,
):
    """
    仕様:
      - 端の NaN は除外し，さらに最初の1年/最後の1年を除外
      - 閾値: El Niño は smoothed > +std，La Niña は smoothed < -std
      - イベント: 閾値を 90 ステップ以上（= 3ヶ月×30）連続で満たす True 区間
      - ピーク:
          * "max_only": 各イベントのグローバル最大の台地中央のみ
          * "all_peaks": 各イベント内の全ての局所極大の台地中央
      - 出力: day-of-year (1..360) ごとのピーク検出回数（長さ360の配列）
    """
    # 1) smoothed と years のアライン（NaN 除去）
    s = smoothed_series.values
    y = years
    valid = ~np.isnan(s)
    s = s[valid]
    y = y[valid]

    # 2) 最初の1年/最後の1年を除外
    #    years は年単位の連続時刻なので，[1, t_max-1] に制限
    t_min, t_max = np.floor(np.min(y)), np.ceil(np.max(y))
    logger.debug(f"Data range after NaN removal: {t_min:.2f} to {t_max:.2f} years")
    y_mask = (y >= (t_min + 1)) & (y <= (t_max - 1))
    s = s[y_mask]
    y = y[y_mask]

    if s.size == 0:
        return np.zeros(360, dtype=int)

    # 3) 条件マスク（El Niño / La Niña）
    if mode.lower() == "elnino":
        cond = s > (+std_value)
    elif mode.lower() == "lanina":
        cond = s < (-std_value)
    else:
        raise ValueError("mode must be 'elnino' or 'lanina'")

    # 4) 連続 True 区間（イベント）を抽出し，長さ >= 90 のみ残す
    runs = _find_runs_bool(cond)
    runs = [(i0, i1) for (i0, i1) in runs if (i1 - i0 + 1) >= (3 * n_per_month)]

    # 5) 各イベントでピーク検出
    peak_indices_global = []  # smoothed 配列上のグローバル添字
    for i0, i1 in runs:
        seg = s[i0 : i1 + 1]

        if peak_mode.lower() == "max_only":
            c = _plateau_center_global_max(seg)
            if c is not None:
                peak_indices_global.append(i0 + c)
        elif peak_mode.lower() == "all_peaks":
            local_centers = _plateau_centers_local_maxima(seg)
            if len(local_centers) == 0:
                # まれに完全に単調/完全平坦（局所極大が定義できない）なら，台地中央1つ
                c = _plateau_center_global_max(seg)
                if c is not None:
                    peak_indices_global.append(i0 + c)
            else:
                peak_indices_global.extend([i0 + c for c in local_centers])
        else:
            raise ValueError("peak_mode must be 'max_only' or 'all_peaks'")

    if len(peak_indices_global) == 0:
        return np.zeros(360, dtype=int)

    # 6) day-of-year 1..360 に集計
    peak_years = y[np.array(peak_indices_global, dtype=int)]
    days = _fractional_day_of_year(
        peak_years, days_per_year=12 * n_per_month
    )  # 360日暦
    assert 1 <= np.min(days) and np.max(days) <= 12 * n_per_month

    counts = np.bincount(days, minlength=(12 * n_per_month + 1))  # index 0 未使用
    assert counts[0] == 0  # dummy for day 0 in bincount
    counts = counts[1:]  # 1..360

    if not debug:
        return counts.astype(int)
    else:
        return counts.astype(int), peak_indices_global, s, y


def detect_peak_counts_scipy(
    years: np.ndarray,
    smoothed_series: pd.Series,
    std_value: float,
    peak_min_width: int,
    peak_min_distance: int,
    threshold_n_months: int,
    n_per_month: int = 30,
    mode: Literal["elnino", "lanina"] = "elnino",
    peak_mode: Literal["max_only", "all_peaks"] = "max_only",
    debug: bool = False,
):
    """
    SciPy find_peaks を用いた別実装。
    仕様は detect_peak_counts と同じ：
      - 端の NaN 除去 + 最初/最後の 1 年を除外
      - 閾値: El Niño は smoothed > +std, La Niña は smoothed < -std
      - イベント: 閾値を 90 ステップ以上（3ヶ月×30）連続で満たす True 区間
      - ピーク:
        * "max_only": 各イベントのグローバル最大の台地中央のみ
        * "all_peaks": 各イベント内の全局所極大（台地含む）の中央
      - day-of-year 1..360 で検出回数を返す
    """
    # 1) smoothed と years のアライン（NaN 除去）
    s = smoothed_series.values
    y = years
    valid = ~np.isnan(s)
    s = s[valid]
    y = y[valid]

    # 2) 最初/最後の 1 年を除外
    t_min, t_max = np.floor(np.min(y)), np.ceil(np.max(y))
    logger.debug(f"Data range after NaN removal: {t_min:.2f} to {t_max:.2f} years")
    y_mask = (y >= (t_min + 1)) & (y <= (t_max - 1))
    s = s[y_mask]
    y = y[y_mask]

    if s.size == 0:
        return np.zeros(12 * n_per_month, dtype=int)

    # 3) 閾値マスク
    if mode.lower() == "elnino":
        cond = s > (+std_value)
        seg_for_peaks = lambda seg: seg  # 最大をそのまま検出
    elif mode.lower() == "lanina":
        cond = s < (-std_value)
        seg_for_peaks = lambda seg: -seg  # 最小→最大に反転して検出
    else:
        raise ValueError("mode must be 'elnino' or 'lanina'")

    # 4) True 連続区間（イベント）抽出（90 以上）
    runs = _find_runs_bool(cond)
    runs = [
        (i0, i1)
        for (i0, i1) in runs
        if (i1 - i0 + 1) >= (threshold_n_months * n_per_month)
    ]

    peak_indices_global = []

    for i0, i1 in runs:
        seg = s[i0 : i1 + 1]

        # find_peaks: plateau（平坦極大）も拾うため plateau_size を指定
        # 返り値 properties には（SciPyバージョンにより）plateauの端情報が含まれる。
        # それがない場合はピーク添字自身を使う。
        vals = seg_for_peaks(seg)
        peaks, props = scipy.signal.find_peaks(
            vals,
            width=peak_min_width,
            distance=peak_min_distance,
            plateau_size=1,
        )

        # 台地がある場合の中心を計算
        left_edges = props.get("left_edges", None)
        right_edges = props.get("right_edges", None)

        if peaks.size == 0:
            # 局所極大が見つからない（全平坦/単調）の場合は，区間のグローバル最大の台地中心を 1 つ
            if peak_mode.lower() in ("all_peaks", "max_only"):
                vmax = np.max(seg) if mode.lower() == "elnino" else np.min(seg)
                eq = np.where(seg == vmax)[0]
                c_local = int(np.round((eq[0] + eq[-1]) / 2.0))
                peak_indices_global.append(i0 + c_local)
            continue

        # 各ピークの「台地中央」をローカル添字で求める
        centers_local = []
        for k, pk in enumerate(peaks):
            if left_edges is not None and right_edges is not None:
                le = int(left_edges[k])
                re = int(right_edges[k])
                c = int(np.round((le + re) / 2.0))
                centers_local.append(c)
            else:
                # plateau 情報がない場合：同値台地を自力で広げる（ε=0）
                # pk を中心に左/右へ同値連を広げる
                l = pk
                r = pk
                while l - 1 >= 0 and seg[l - 1] == seg[pk]:
                    l -= 1
                while r + 1 < len(seg) and seg[r + 1] == seg[pk]:
                    r += 1
                c = int(np.round((l + r) / 2.0))
                centers_local.append(c)

        if peak_mode.lower() == "all_peaks":
            for c_local in centers_local:
                peak_indices_global.append(i0 + c_local)

        elif peak_mode.lower() == "max_only":
            # イベント内のグローバル最大（La Niña は最小）に最も近い台地中心を 1 つ選ぶ
            target_val = np.max(seg) if mode.lower() == "elnino" else np.min(seg)
            # 台地中心の値（元の seg）と target との距離が最小のもの
            diffs = [abs(seg[c] - target_val) for c in centers_local]
            j = int(np.argmin(diffs))
            peak_indices_global.append(i0 + centers_local[j])

        else:
            raise ValueError("peak_mode must be 'max_only' or 'all_peaks'")

    if len(peak_indices_global) == 0:
        return np.zeros(12 * n_per_month, dtype=int)

    # 6) day-of-year へ集計（360日暦）
    peak_years = y[np.array(peak_indices_global, dtype=int)]
    days = _fractional_day_of_year(peak_years, days_per_year=12 * n_per_month)
    assert 1 <= np.min(days) and np.max(days) <= 12 * n_per_month

    counts = np.bincount(days, minlength=(12 * n_per_month + 1))
    assert counts[0] == 0  # dummy for day 0 in bincount
    counts = counts[1:]  # 1..360

    if not debug:
        return counts.astype(int)
    else:
        return counts.astype(int), peak_indices_global, s, y
