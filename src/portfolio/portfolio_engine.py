from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time

import numpy as np

from configs.schema import AlphaConfig, RiskConfig
from src.core.models import MarketSeries
from src.utils.math import clamp


def _coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


@dataclass
class PortfolioSnapshot:
    """投资组合快照"""
    target_weights: Dict[str, float]
    selected: List[str]
    volatilities: Dict[str, float]
    notes: str = ""
    entry_candidates: List[str] = field(default_factory=list)


class PortfolioEngine:
    """投资组合引擎

    根据Alpha评分分配目标权重。
    支持轻量级 Qlib-inspired optimizer：在当前权重与上一轮权重之间做平滑，
    以降低换手抖动和边界来回交易。
    """
    
    def __init__(self, alpha_cfg: AlphaConfig, risk_cfg: RiskConfig):
        """初始化投资组合引擎
        
        Args:
            alpha_cfg: Alpha配置
            risk_cfg: 风险配置
        """
        self.alpha_cfg = alpha_cfg
        self.risk_cfg = risk_cfg
        self.run_id = ""

    def set_run_id(self, run_id: Optional[str]) -> None:
        self.run_id = str(run_id or "").strip()

    def _strategy_signals_path(self) -> Optional[Path]:
        if not self.run_id:
            return None
        return Path("reports") / "runs" / self.run_id / "strategy_signals.json"

    def _load_optimizer_state(self) -> Dict[str, Any]:
        try:
            p = Path(str(getattr(self.alpha_cfg, 'optimizer_state_path', 'reports/portfolio_optimizer_state.json')))
            if not p.exists():
                return {'weights': {}, 'updated_ts': 0}
            obj = json.loads(p.read_text(encoding='utf-8'))
            if isinstance(obj, dict):
                w = obj.get('weights') or {}
                if isinstance(w, dict):
                    return {
                        'weights': {str(k): float(v) for k, v in w.items() if isinstance(v, (int, float))},
                        'updated_ts': int(obj.get('updated_ts', 0) or 0),
                    }
        except Exception:
            pass
        return {'weights': {}, 'updated_ts': 0}

    def _save_optimizer_state(self, weights: Dict[str, float]) -> None:
        try:
            p = Path(str(getattr(self.alpha_cfg, 'optimizer_state_path', 'reports/portfolio_optimizer_state.json')))
            p.parent.mkdir(parents=True, exist_ok=True)
            obj = {
                'weights': {str(k): float(v) for k, v in (weights or {}).items()},
                'updated_ts': int(time.time()),
            }
            tmp = p.with_suffix(p.suffix + '.tmp')
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(p)
        except Exception:
            pass

    def _load_topk_state(self) -> Dict[str, Any]:
        try:
            cfg = getattr(self.alpha_cfg, "topk_dropout", None)
            if not cfg:
                return {"selected": [], "hold_cycles": {}, "updated_ts": 0}
            p = Path(str(getattr(cfg, "state_path", "reports/topk_dropout_state.json")))
            if not p.exists():
                return {"selected": [], "hold_cycles": {}, "updated_ts": 0}
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj.setdefault("selected", [])
                obj.setdefault("hold_cycles", {})
                obj.setdefault("updated_ts", 0)
                return obj
        except Exception:
            pass
        return {"selected": [], "hold_cycles": {}, "updated_ts": 0}

    def _save_topk_state(self, selected: List[str], hold_cycles: Dict[str, int]) -> None:
        try:
            cfg = getattr(self.alpha_cfg, "topk_dropout", None)
            if not cfg:
                return
            p = Path(str(getattr(cfg, "state_path", "reports/topk_dropout_state.json")))
            p.parent.mkdir(parents=True, exist_ok=True)
            obj = {
                "selected": list(selected or []),
                "hold_cycles": {k: int(v) for k, v in (hold_cycles or {}).items()},
                "updated_ts": int(time.time()),
            }
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception:
            pass

    @staticmethod
    def _sort_symbols_by_priority(
        symbols: List[str],
        *,
        selection_scores: Dict[str, float],
        preferred: Optional[List[str]] = None,
    ) -> List[str]:
        preferred = list(preferred or [])
        preferred_set = set(preferred)
        preferred_rank = {sym: idx for idx, sym in enumerate(preferred)}
        uniq_symbols = list(dict.fromkeys(symbols or []))
        return sorted(
            uniq_symbols,
            key=lambda sym: (
                0 if sym in preferred_set else 1,
                -float(selection_scores.get(sym, -1e9)),
                preferred_rank.get(sym, len(preferred_rank) + 10_000),
                str(sym),
            ),
        )

    def _apply_topk_dropout(
        self,
        selected: List[str],
        selection_scores: Dict[str, float],
        audit: Optional[Any] = None,
        target_k: Optional[int] = None,
    ) -> List[str]:
        """TopkDropout: 每轮最多替换 n_drop，且仅替换满足最短持有轮次的旧标的。"""
        cfg = getattr(self.alpha_cfg, "topk_dropout", None)
        if not cfg or not bool(getattr(cfg, "enabled", False)):
            return selected

        n_drop = max(1, int(getattr(cfg, "n_drop_per_cycle", 2) or 2))
        hold_req = max(1, int(getattr(cfg, "hold_cycles", 2) or 2))

        st = self._load_topk_state()
        prev_selected = [s for s in (st.get("selected") or []) if isinstance(s, str)]
        prev_hold = st.get("hold_cycles") or {}
        target_k = int(target_k or len(selected) or 0)

        # 冷启动
        if not prev_selected:
            selected_sorted = self._sort_symbols_by_priority(
                list(selected or []),
                selection_scores=selection_scores,
                preferred=list(selected or []),
            )[:target_k]
            hold_new = {s: 1 for s in selected_sorted}
            self._save_topk_state(selected_sorted, hold_new)
            return selected_sorted

        candidate = list(selected)
        candidate_set = set(candidate)
        prev_set = set(prev_selected)

        newcomers = [s for s in candidate if s not in prev_set]
        hold_cont = [s for s in prev_selected if s in candidate_set]

        # 仅允许替换满足 hold_req 的旧持仓
        droppable = [s for s in prev_selected if int(prev_hold.get(s, 1)) >= hold_req]
        # 在 droppable 里优先淘汰当前评分最低者
        droppable_sorted = sorted(droppable, key=lambda s: float(selection_scores.get(s, -1e9)))

        max_replace = min(n_drop, len(newcomers), len(droppable_sorted))
        drop_set = set(droppable_sorted[:max_replace])
        add_list = newcomers[:max_replace]

        kept = [s for s in prev_selected if s not in drop_set]
        merged = kept + [s for s in add_list if s not in kept]

        # 用候选池补齐（避免容量不足）
        for s in candidate:
            if s not in merged:
                merged.append(s)

        # 保持与候选同样容量
        merged = self._sort_symbols_by_priority(
            merged,
            selection_scores=selection_scores,
            preferred=candidate,
        )[:target_k]

        # 更新 hold cycle
        hold_new: Dict[str, int] = {}
        for s in merged:
            if s in prev_set and s in hold_cont:
                hold_new[s] = int(prev_hold.get(s, 1)) + 1
            else:
                hold_new[s] = 1

        self._save_topk_state(merged, hold_new)

        if audit:
            audit.add_note(
                "TopkDropout applied: prev={} cand={} kept={} replace={} hold_req={} n_drop={}"
                .format(len(prev_selected), len(candidate), len(merged), max_replace, hold_req, n_drop)
            )

        return merged

    def _load_fused_signals(self) -> Optional[Dict[str, float]]:
        """Load fused signals from the current run if available."""
        try:
            strategy_file = self._strategy_signals_path()
            if strategy_file is None or not strategy_file.exists():
                return None

            with open(strategy_file, encoding='utf-8') as f:
                data = json.load(f)
                fused = data.get("fused", {})
                if fused:
                    scores = {}
                    for sym, sig in fused.items():
                        direction = sig.get("direction", "hold")
                        score = sig.get("score", 0)
                        if direction == "buy":
                            scores[sym] = float(score)
                        elif direction == "sell":
                            scores[sym] = -float(score)
                        else:
                            scores[sym] = 0.0
                    return scores if scores else None
        except Exception:
            pass
        return None

    @staticmethod
    def _merge_fused_scores_with_scores(
        fused_scores: Optional[Dict[str, float]],
        scores: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        if not fused_scores:
            return scores if scores else None
        merged = {str(sym): float(val) for sym, val in (fused_scores or {}).items()}
        for sym, raw_base in (scores or {}).items():
            base = float(raw_base)
            if sym not in merged:
                merged[sym] = base
                continue

            fused = float(merged[sym])
            if fused >= 0.0 and base >= 0.0:
                merged[sym] = min(fused, base)
            elif fused <= 0.0 and base <= 0.0:
                merged[sym] = max(fused, base)
            else:
                merged[sym] = base
        return merged

    def _get_dynamic_max_positions(self) -> Optional[int]:
        """Return effective max positions cap.

        Priority:
        1) risk.max_positions_override (hard override)
        2) auto-risk level mapping from reports/auto_risk_eval.json
        """
        try:
            ov = getattr(self.risk_cfg, "max_positions_override", None)
            if ov is not None:
                ov_i = int(ov)
                if ov_i > 0:
                    return ov_i
        except Exception:
            pass

        try:
            p = Path("reports/auto_risk_eval.json")
            if not p.exists():
                return None
            obj = json.loads(p.read_text(encoding="utf-8"))
            lvl = str(obj.get("current_level", "")).upper()
            cap_map = {
                "PROTECT": 1,
                "DEFENSE": 3,
                "NEUTRAL": 5,
                "ATTACK": 8,
            }
            return cap_map.get(lvl)
        except Exception:
            return None

    def allocate(
        self,
        scores: Dict[str, float],
        market_data: Dict[str, MarketSeries],
        regime_mult: float,
        audit: Optional[Any] = None,
    ) -> PortfolioSnapshot:
        """分配目标权重
        
        Args:
            scores: Alpha评分 {symbol: score}
            market_data: 市场数据
            regime_mult: 市场状态乘数
            audit: 审计对象
            
        Returns:
            投资组合快照
        """
        # Try to use fused signals from multi-strategy if available
        fused_scores = self._load_fused_signals()
        if fused_scores:
            selection_scores = self._merge_fused_scores_with_scores(fused_scores, scores)
            if audit:
                audit.add_note(
                    f"Using fused signals for selection with alpha adjustments: {list(fused_scores.keys())[:5]}"
                )
        else:
            selection_scores = scores
        
        if not selection_scores:
            return PortfolioSnapshot(target_weights={}, selected=[], volatilities={}, notes="no_scores", entry_candidates=[])

        # Select top pct by score (using fused signals if available)
        items = sorted(selection_scores.items(), key=lambda kv: float(kv[1]), reverse=True)

        # top-k size from pct / override
        k_pct = max(1, int(np.ceil(len(items) * float(self.alpha_cfg.long_top_pct))))
        k = k_pct
        try:
            topk_cfg = getattr(self.alpha_cfg, "topk_dropout", None)
            topk_override = int(getattr(topk_cfg, "topk_override", 0) or 0) if topk_cfg else 0
            if topk_override > 0:
                k = min(len(items), max(1, topk_override))
        except Exception:
            pass

        selected = [s for s, score in items[:k] if score >= float(getattr(self.alpha_cfg, "min_score_threshold", 0.0))]
        max_pos = self._get_dynamic_max_positions()
        entry_candidates = list(selected)
        if max_pos is not None and max_pos >= 0 and len(entry_candidates) > max_pos:
            entry_candidates = self._sort_symbols_by_priority(
                entry_candidates,
                selection_scores=selection_scores,
                preferred=entry_candidates,
            )[:max_pos]

        # TopkDropout limited replacement（在风险cap之前，先控换手）
        selected = self._apply_topk_dropout(
            selected,
            selection_scores=selection_scores,
            audit=audit,
            target_k=min(len(selected), int(max_pos)) if max_pos is not None and max_pos >= 0 else len(selected),
        )
        selected = self._sort_symbols_by_priority(
            selected,
            selection_scores=selection_scores,
            preferred=entry_candidates or selected,
        )

        # Enforce dynamic risk-level position cap (PROTECT/DEFENSE/NEUTRAL/ATTACK)
        if max_pos is not None and max_pos >= 0 and len(selected) > max_pos:
            selected = selected[:max_pos]
            if audit:
                audit.add_note(f"AutoRisk position cap applied: max_positions={max_pos}")

        # For weight calculation:
        # - default: use fused scores when fused selection is active (to keep selection/sizing consistent)
        # - fallback: use original alpha scores
        use_fused_for_weighting = bool(getattr(self.alpha_cfg, 'use_fused_score_for_weighting', True))
        if fused_scores and use_fused_for_weighting:
            weights_scores = self._merge_fused_scores_with_scores(fused_scores, scores)
            if audit:
                audit.add_note("Using fused scores for weighting with alpha adjustments")
        else:
            weights_scores = scores if scores else fused_scores
        
        # Handle case where selected symbols may not be in weights_scores
        # (e.g., fused signals include symbols not in alpha scores)
        valid_selected = [s for s in selected if s in weights_scores]
        if len(valid_selected) != len(selected) and audit:
            skipped = [s for s in selected if s not in weights_scores]
            audit.add_note(f"Skipping symbols not in scores: {skipped}")
        selected = valid_selected
        
        if not selected:
            return PortfolioSnapshot(
                target_weights={},
                selected=[],
                volatilities={},
                notes="no_valid_selection",
                entry_candidates=entry_candidates,
            )
        
        vols: Dict[str, float] = {}
        inv: Dict[str, float] = {}
        for sym in selected:
            s = market_data.get(sym)
            if not s or len(s.close) < 2:
                vols[sym] = 1.0
                inv[sym] = 1.0
                continue
            c = np.array(s.close, dtype=float)
            rets = np.diff(c) / c[:-1]
            # window ~ 20d on 1h bars
            w = min(len(rets), 24 * 20)
            rv = float(np.std(rets[-w:])) if w > 10 else float(np.std(rets))
            rv = max(rv, 1e-6)
            vols[sym] = rv
            inv[sym] = 1.0 / rv

        inv_sum = float(sum(inv.values())) or 1.0
        base_w = {sym: float(inv[sym]) / inv_sum for sym in selected}

        # Confidence weighting: softmax with temperature (更平滑的映射)
        sel_scores = np.array([weights_scores[s] for s in selected], dtype=float)
        
        # 方法1: softmax with temperature (避免0权重)
        # temperature 参数优化：0.5→0.9 降低集中度，减少换手
        # 目标 Effective N (1/∑w²) 在 3-6 之间
        temperature = 0.9  # 温度参数，越小越集中，越大越分散
        exp_scores = np.exp(sel_scores / temperature)
        softmax_probs = exp_scores / np.sum(exp_scores)
        conf = {s: float(softmax_probs[i]) for i, s in enumerate(selected)}
        
        # 方法2: 带下限的min-max (保留原逻辑作为备选)
        # mn = float(np.min(sel_scores))
        # mx = float(np.max(sel_scores))
        # denom = (mx - mn) if (mx - mn) != 0 else 1.0
        # conf = {s: max(0.2, float((scores[s] - mn) / denom)) for s in selected}  # 20%下限
        
        # 诊断：记录为什么conf可能为0
        portfolio_debug = {
            "inv_vol_norm": base_w,
            "confidence_raw": conf,
            "score_stats": {
                "min": float(np.min(sel_scores)) if len(sel_scores) > 0 else 0.0,
                "max": float(np.max(sel_scores)) if len(sel_scores) > 0 else 0.0,
                "std": float(np.std(sel_scores)) if len(sel_scores) > 1 else 0.0,
                "count": len(sel_scores)
            }
        }
        
        # 如果所有confidence都是0，fallback到等权
        all_conf_zero = all(abs(c) < 1e-12 for c in conf.values())
        if all_conf_zero and len(selected) > 0:
            # Fallback: 等权分配
            fallback_weight = 1.0 / len(selected)
            conf = {s: fallback_weight for s in selected}
            portfolio_debug["fallback_reason"] = "all_confidence_zero"
            portfolio_debug["fallback_weight"] = fallback_weight

        raw = {s: base_w[s] * conf[s] for s in selected}
        raw_sum = float(sum(raw.values())) or 1.0
        w2 = {s: raw[s] / raw_sum for s in selected}

        # Qlib-inspired lightweight optimizer:
        # blend current weights with previous cycle to reduce churn.
        if bool(getattr(self.alpha_cfg, 'optimizer_enabled', False)):
            try:
                state = self._load_optimizer_state()
                prev_w = state.get('weights') or {}
                lam = float(_coalesce(getattr(self.alpha_cfg, 'optimizer_prev_weight_penalty', 0.35), 0.35))
                lam = clamp(lam, 0.0, 1.0)
                floor = float(getattr(self.alpha_cfg, 'optimizer_min_weight_floor', 0.0) or 0.0)
                floor = clamp(floor, 0.0, 0.2)

                merged_syms = sorted(set(list(w2.keys()) + list(prev_w.keys())))
                blended = {}
                for s in merged_syms:
                    cur = float(w2.get(s, 0.0))
                    prev = float(prev_w.get(s, 0.0))
                    v = (1.0 - lam) * cur + lam * prev
                    if s in w2 and v < floor:
                        v = floor
                    blended[s] = max(0.0, v)

                # normalize only selected symbols to keep target basket explicit
                sel_sum = float(sum(blended.get(s, 0.0) for s in selected)) or 1.0
                w2 = {s: float(blended.get(s, 0.0)) / sel_sum for s in selected}

                if audit:
                    audit.add_note(
                        f"Optimizer applied: lambda={lam:.2f}, floor={floor:.3f}, prev_n={len(prev_w)}"
                    )
            except Exception as e:
                if audit:
                    audit.add_note(f"Optimizer skipped: {e}")

        # 记录zero_reason_by_symbol
        zero_reason_by_symbol = {}
        for s in selected:
            if abs(w2[s]) < 1e-12:
                if abs(base_w[s]) < 1e-12:
                    zero_reason_by_symbol[s] = "inv_vol_zero"
                elif abs(conf[s]) < 1e-12:
                    zero_reason_by_symbol[s] = "confidence_zero"
                else:
                    zero_reason_by_symbol[s] = "normalization_zero"

        portfolio_debug["zero_reason_by_symbol"] = zero_reason_by_symbol
        portfolio_debug["weight_pre_clip"] = w2

        # Apply regime multiplier to overall gross exposure, then cap max_single_weight
        gross = float(self.risk_cfg.max_gross_exposure) * float(regime_mult)
        gross = clamp(gross, 0.0, float(self.risk_cfg.max_gross_exposure))

        capped = {s: min(w2[s] * gross, float(self.risk_cfg.max_single_weight)) for s in selected}
        
        # 将portfolio_debug添加到audit
        if audit is not None and hasattr(audit, 'portfolio_debug'):
            audit.portfolio_debug = portfolio_debug
            # 记录portfolio_rejects
            if zero_reason_by_symbol:
                for reason in zero_reason_by_symbol.values():
                    if hasattr(audit, 'reject'):
                        audit.reject(f"portfolio_{reason}")

        # persist final target weights for next-cycle optimizer smoothing
        if bool(getattr(self.alpha_cfg, 'optimizer_enabled', False)):
            self._save_optimizer_state(capped)

        return PortfolioSnapshot(
            target_weights=capped,
            selected=selected,
            volatilities=vols,
            entry_candidates=entry_candidates,
        )

    def scale_targets(self, targets: Dict[str, float], mult: float) -> Dict[str, float]:
        """Scale portfolio target weights by exposure multiplier, keeping caps."""
        m = float(mult)
        if m >= 1.0:
            return dict(targets or {})
        out: Dict[str, float] = {}
        for sym, w in (targets or {}).items():
            w2 = float(w) * m
            out[sym] = min(w2, float(self.risk_cfg.max_single_weight))
        return out
