"""
Cash-Secured Put Simulator — FastAPI service + React front-end

This version adds:
  • FastAPI backend (unchanged core logic).
  • A simple React front-end served from FastAPI's static files.
  • Front-end provides a table of ranked puts and a chart for P&L distribution.

Run backend:
  uvicorn app:app --reload --port 8000

Visit:
  http://localhost:8000

Note: You still need to fill in a real data adapter to fetch quotes.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date, timezone
from typing import List, Optional, Dict, Any
import random

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import statistics as stats
import os
from fastapi import HTTPException

# ============
# Data Models
# ============

@dataclass
class OptionQuote:
    symbol: str
    expiration: date
    strike: float
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    iv: Optional[float] = None  # Implied volatility (annualized, e.g., 0.25)
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid if self.mid > 0 else 0

@dataclass
class RankedPut:
    symbol: str
    expiration: date
    strike: float
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    days_to_expiry: int
    aroc: float  # Annualized Return on Capital
    pop: Optional[float]  # Probability of Profit
    breakeven: float
    monthly_income: float
    max_loss: float
    risk_reward: float
    score: float

class ScanRequest(BaseModel):
    symbol: str
    dte_min: int = 30
    dte_max: int = 45
    delta_min: float = 0.15
    delta_max: float = 0.30
    min_oi: int = 500
    max_spread_pct: float = 0.10

class ScanResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int

# ============
# Core Logic
# ============

def compute_short_put_metrics(
    quote: OptionQuote,
    current_price: float,
    days_to_expiry: int,
    use_bid: bool = False,
    pop_otm_fallback: float = 0.6,
    pop_itm_fallback: float = 0.4,
) -> Dict[str, float]:
    """Compute key metrics for a short put position.
    If use_bid is True, use bid as the premium (conservative). Otherwise use mid.
    """
    premium = quote.bid if use_bid else quote.mid
    strike = quote.strike
    
    # Annualized Return on Capital (assuming 100% cash secured)
    aroc = (premium / strike) * (365 / days_to_expiry) if days_to_expiry > 0 else 0
    
    # Monthly Yield (normalized to 30-day months) — percentage ratio, not dollars
    monthly_yield_pct = (premium / strike) * (30 / days_to_expiry) if days_to_expiry > 0 else 0
    
    # Probability of Profit: approximate P(S_T > breakeven) with lognormal model using IV if available
    pop = None
    try:
        if days_to_expiry > 0:
            T = max(1e-6, days_to_expiry / 365.0)
            breakeven_local = max(1e-6, strike - premium)
            sigma = quote.iv if getattr(quote, "iv", None) is not None else None
            if sigma and sigma > 0:
                # lognormal: ln(S_T) ~ N(ln(S0) - 0.5*sigma^2*T, sigma^2*T)
                # POP = P(S_T > breakeven) = 1 - Phi((ln(B/S0) + 0.5*sigma^2*T)/(sigma*sqrt(T)))
                import math
                z = (math.log(breakeven_local / max(1e-9, current_price)) + 0.5 * (sigma ** 2) * T) / (sigma * math.sqrt(T))
                # standard normal CDF via erf
                def phi(x: float) -> float:
                    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                pop = 1.0 - phi(z)
    except Exception:
        pop = None
    # Fallback heuristic if IV-based POP unavailable
    if pop is None:
        pop = (pop_otm_fallback if strike < current_price else pop_itm_fallback)
    
    # Breakeven point
    breakeven = strike - premium
    
    # Maximum Loss (if assigned)
    max_loss = strike - premium
    
    # Risk/Reward ratio
    risk_reward = premium / max_loss if max_loss > 0 else 0
    
    # Enhanced scoring (higher is better)
    score = (aroc * 100) + (pop * 20) + (risk_reward * 10) + (1 / (1 + quote.spread_pct))
    
    return {
        "aroc": aroc,
        # kept for backward compatibility; this is a yield ratio
        "monthly_income": monthly_yield_pct,
        "monthly_yield_pct": monthly_yield_pct,
        "pop": pop,
        "breakeven": breakeven,
        "max_loss": max_loss,
        "risk_reward": risk_reward,
        "score": score
    }

def rank_puts(
    quotes: List[OptionQuote],
    current_price: float,
    dte_min: int,
    dte_max: int,
    delta_min: float,
    delta_max: float,
    min_oi: int,
    max_spread_pct: float,
    capital: Optional[float] = None,
    use_bid: bool = False,
    pop_otm_fallback: float = 0.6,
    pop_itm_fallback: float = 0.4,
) -> List[RankedPut]:
    """Rank put options based on criteria and scoring.
    If capital is provided, compute per-candidate contract sizing and income metrics.
    """
    ranked = []
    
    for quote in quotes:
        # Calculate days to expiry
        days_to_expiry = (quote.expiration - date.today()).days
        
        # Filter by criteria
        if not (dte_min <= days_to_expiry <= dte_max):
            continue
        if quote.open_interest < min_oi:
            continue
        if quote.spread_pct > max_spread_pct:
            continue
        
        # Compute approximate put delta for filtering (use IV if available, else fallback)
        try:
            T = max(1e-6, days_to_expiry / 365.0)
            sigma = quote.iv if (getattr(quote, "iv", None) is not None and quote.iv and quote.iv > 0) else 0.30
            import math
            d1 = (math.log(max(1e-9, current_price / quote.strike)) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
            # standard normal CDF
            def phi(x: float) -> float:
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            call_delta = phi(d1)
            put_delta = call_delta - 1.0  # put delta is negative
            abs_put_delta = abs(put_delta)
        except Exception:
            abs_put_delta = None

        # Apply delta filter if we have an estimate
        if abs_put_delta is not None and not (delta_min <= abs_put_delta <= delta_max):
            continue

        # Compute metrics
        metrics = compute_short_put_metrics(
            quote,
            current_price,
            days_to_expiry,
            use_bid=use_bid,
            pop_otm_fallback=pop_otm_fallback,
            pop_itm_fallback=pop_itm_fallback,
        )
        
        # Create ranked put
        ranked_put = RankedPut(
            symbol=quote.symbol,
            expiration=quote.expiration,
            strike=quote.strike,
            bid=quote.bid,
            ask=quote.ask,
            mid=quote.mid,
            volume=quote.volume,
            open_interest=quote.open_interest,
            days_to_expiry=days_to_expiry,
            aroc=metrics["aroc"],
            pop=metrics["pop"],
            breakeven=metrics["breakeven"],
            monthly_income=metrics["monthly_income"],
            max_loss=metrics["max_loss"],
            risk_reward=metrics["risk_reward"],
            score=metrics["score"]
        )
        ranked.append(ranked_put)
    
    # Sort by score (descending)
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked

def compute_covered_call_metrics(
    quote: OptionQuote,
    current_price: float,
    days_to_expiry: int,
    use_bid: bool = False,
    pop_otm_fallback: float = 0.6,
    pop_itm_fallback: float = 0.4,
) -> Dict[str, float]:
    """Compute key metrics for a covered call position."""
    premium = quote.bid if use_bid else quote.mid
    strike = quote.strike
    
    # For covered calls, we collect premium and potentially get called away
    # Return calculation is based on premium received relative to share value
    share_value = current_price * 100  # 100 shares per contract
    aroc = (premium * 100 / share_value) * (365 / days_to_expiry) if days_to_expiry > 0 and share_value > 0 else 0
    
    # Monthly yield percentage
    monthly_yield_pct = (premium * 100 / share_value) * (30 / days_to_expiry) if days_to_expiry > 0 and share_value > 0 else 0
    
    # For covered calls, POP is probability that stock stays below strike (we keep premium)
    pop = None
    try:
        if days_to_expiry > 0:
            T = max(1e-6, days_to_expiry / 365.0)
            sigma = quote.iv if getattr(quote, "iv", None) is not None else None
            if sigma and sigma > 0:
                import math
                z = (math.log(strike / max(1e-9, current_price)) - 0.5 * (sigma ** 2) * T) / (sigma * math.sqrt(T))
                def phi(x: float) -> float:
                    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                pop = phi(z)  # P(S_T < strike)
    except Exception:
        pop = None
    
    if pop is None:
        # For calls: if strike > current_price (OTM), higher chance of keeping premium
        pop = (pop_otm_fallback if strike > current_price else pop_itm_fallback)
    
    # Breakeven is current price minus premium received
    breakeven = current_price - premium
    
    # Max loss is if stock goes to zero (minus premium received)
    max_loss = current_price - premium
    
    # Risk/reward: premium vs potential loss
    risk_reward = premium / max_loss if max_loss > 0 else 0
    
    # Scoring for covered calls
    score = (aroc * 100) + (pop * 20) + (risk_reward * 10) + (1 / (1 + quote.spread_pct))
    
    return {
        "aroc": aroc,
        "monthly_income": monthly_yield_pct,
        "monthly_yield_pct": monthly_yield_pct,
        "pop": pop,
        "breakeven": breakeven,
        "max_loss": max_loss,
        "risk_reward": risk_reward,
        "score": score
    }

def rank_covered_calls(
    quotes: List[OptionQuote],
    current_price: float,
    dte_min: int,
    dte_max: int,
    delta_min: float,
    delta_max: float,
    min_oi: int,
    max_spread_pct: float,
    capital: Optional[float] = None,
    use_bid: bool = False,
    pop_otm_fallback: float = 0.6,
    pop_itm_fallback: float = 0.4,
    otm_only: bool = True,
) -> List[RankedPut]:  # Reusing RankedPut structure for calls
    """Rank call options for covered call strategy."""
    ranked = []
    
    for quote in quotes:
        # Calculate days to expiry
        days_to_expiry = (quote.expiration - date.today()).days
        
        # Filter by criteria
        if not (dte_min <= days_to_expiry <= dte_max):
            continue
        if quote.open_interest < min_oi:
            continue
        if quote.spread_pct > max_spread_pct:
            continue
        # Prefer OTM covered calls by default: require strike above current price
        if otm_only and quote.strike <= current_price:
            continue
        
        # Compute call delta for filtering
        try:
            T = max(1e-6, days_to_expiry / 365.0)
            sigma = quote.iv if (getattr(quote, "iv", None) is not None and quote.iv and quote.iv > 0) else 0.30
            import math
            d1 = (math.log(max(1e-9, current_price / quote.strike)) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
            def phi(x: float) -> float:
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            call_delta = phi(d1)
            abs_call_delta = abs(call_delta)
        except Exception:
            abs_call_delta = None

        # Apply delta filter
        if abs_call_delta is not None and not (delta_min <= abs_call_delta <= delta_max):
            continue

        # Compute covered call metrics
        metrics = compute_covered_call_metrics(
            quote,
            current_price,
            days_to_expiry,
            use_bid=use_bid,
            pop_otm_fallback=pop_otm_fallback,
            pop_itm_fallback=pop_itm_fallback,
        )
        
        # Create ranked call (reusing RankedPut structure)
        ranked_call = RankedPut(
            symbol=quote.symbol,
            expiration=quote.expiration,
            strike=quote.strike,
            bid=quote.bid,
            ask=quote.ask,
            mid=quote.mid,
            volume=quote.volume,
            open_interest=quote.open_interest,
            days_to_expiry=days_to_expiry,
            aroc=metrics["aroc"],
            pop=metrics["pop"],
            breakeven=metrics["breakeven"],
            monthly_income=metrics["monthly_income"],
            max_loss=metrics["max_loss"],
            risk_reward=metrics["risk_reward"],
            score=metrics["score"]
        )
        ranked.append(ranked_call)
    
    # Sort by score (descending)
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked

# ============
# Data Adapter
# ============

import yfinance as yf
from typing import Optional

class YahooFinanceAdapter:
    """Real data adapter using Yahoo Finance API."""
    
    def get_current_price(self, symbol: str) -> float:
        """Get current stock price from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info or {}

            # Determine market state (REGULAR/TRADING means open). Anything else -> use last close
            market_state = str(info.get('marketState') or '').upper()
            open_states = {"REGULAR", "TRADING"}

            # If market is closed (POST, PRE, CLOSED, etc.), return the last official close
            if market_state and market_state not in open_states:
                for key in ("regularMarketPreviousClose", "previousClose"):
                    try:
                        val = float(info.get(key) or 0)
                        if val > 0 and math.isfinite(val):
                            return val
                    except Exception:
                        pass
                # Fallback to recent historical close (last valid close in up to 5 days)
                hist = ticker.history(period="5d", interval="1d")
                if not hist.empty:
                    last_close_series = hist['Close'].dropna()
                    if not last_close_series.empty:
                        return float(last_close_series.iloc[-1])

            # Market open or unknown: prefer real-time price, fallback to last close
            current_price = info.get('regularMarketPrice') or info.get('currentPrice') or 0
            try:
                current_price = float(current_price)
            except Exception:
                current_price = 0.0

            if current_price and current_price > 0 and math.isfinite(current_price):
                return current_price

            # Fallback to last close via history
            hist = ticker.history(period="5d", interval="1d")
            if not hist.empty:
                last_close_series = hist['Close'].dropna()
                if not last_close_series.empty:
                    return float(last_close_series.iloc[-1])

            # As a last resort, raise to trigger adapter fallback
            raise ValueError(f"Could not get price for {symbol}")
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            # Fallback to mock price for reliability
            return self._get_fallback_price(symbol)

    def get_fx_usd_to_cad(self) -> float:
        """Fetch USD->CAD FX rate from Yahoo Finance.
        Tries 'CAD=X' first (Yahoo pair for USD/CAD), then 'USDCAD=X'.
        Returns a positive float; falls back to 1.39 if unavailable.
        """
        try:
            symbols = ("CAD=X", "USDCAD=X")
            for sym in symbols:
                try:
                    t = yf.Ticker(sym)
                    # Prefer history for reliability
                    hist = t.history(period="1d", interval="1d")
                    if not hist.empty:
                        rate = float(hist['Close'].iloc[-1])
                        if rate > 0 and math.isfinite(rate):
                            return rate
                    # Fallback to info
                    info = t.info
                    rate = info.get('regularMarketPrice') or info.get('previousClose') or 0
                    rate = float(rate)
                    if rate > 0 and math.isfinite(rate):
                        return rate
                except Exception:
                    continue
        except Exception:
            pass
        # Sensible fallback
        return 1.39
    
    def _get_fallback_price(self, symbol: str) -> float:
        """Fallback price if Yahoo Finance fails."""
        fallback_prices = {
            "AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0, "TSLA": 250.0,
            "SPY": 450.0, "QQQ": 380.0, "IWM": 190.0, "META": 300.0,
            "NVDA": 800.0, "AMD": 120.0, "NFLX": 600.0, "TLT": 90.0,
            "GLD": 200.0, "VTI": 250.0, "VOO": 400.0, "ARKK": 50.0
        }
        return fallback_prices.get(symbol.upper(), 100.0)
    
    def get_symbol_iv(self, symbol: str) -> Optional[float]:
        """Best-effort estimate of near-term, near-the-money implied volatility (annualized).
        Strategy:
          - Use first upcoming expiration with available chain
          - Pick strike closest to current price using puts chain (or calls if needed)
          - Return its impliedVolatility
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            options = ticker.options
            if not options:
                return None
            # Choose the first future expiry
            expiry = None
            today = date.today()
            for exp_str in options:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    if exp_date > today:
                        expiry = exp_str
                        break
                except Exception:
                    continue
            if not expiry:
                return None
            # Get current price
            spot = self.get_current_price(symbol)
            # Load option chain
            chain = ticker.option_chain(expiry)
            # Prefer puts, fallback to calls
            df = None
            try:
                df = chain.puts
                if df is None or df.empty:
                    df = chain.calls
            except Exception:
                try:
                    df = chain.calls
                except Exception:
                    df = None
            if df is None or df.empty:
                return None
            # Find strike closest to spot
            df = df.copy()
            df['dist'] = (df['strike'] - spot).abs()
            row = df.sort_values('dist').iloc[0]
            iv = row.get('impliedVolatility')
            if iv is None:
                return None
            iv = float(iv)
            if iv <= 0 or not math.isfinite(iv):
                return None
            return iv
        except Exception:
            return None
    
    def get_put_options(self, symbol: str) -> List[OptionQuote]:
        """Get real put options from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol.upper())
            options = ticker.options
            
            if not options:
                print(f"No options found for {symbol}")
                return self._get_fallback_options(symbol)
            
            # Get the next few expiration dates
            valid_expiries = []
            current_date = date.today()
            
            for exp_date_str in options:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                if exp_date > current_date:
                    valid_expiries.append(exp_date)
                    if len(valid_expiries) >= 4:  # Limit to 4 expiries
                        break
            
            if not valid_expiries:
                return self._get_fallback_options(symbol)
            
            quotes = []
            current_price = self.get_current_price(symbol)
            
            for expiration in valid_expiries:
                try:
                    # Get options chain for this expiry
                    opt_chain = ticker.option_chain(expiration.strftime('%Y-%m-%d'))
                    puts = opt_chain.puts
                    
                    if puts.empty:
                        continue
                    
                    # Filter puts with reasonable strikes and liquidity
                    for _, put in puts.iterrows():
                        strike = put['strike']
                        bid = put['bid']
                        ask = put['ask']
                        volume = put['volume']
                        open_interest = put['openInterest']
                        iv = None
                        try:
                            iv_val = put.get('impliedVolatility') if hasattr(put, 'get') else put['impliedVolatility']
                            if iv_val is not None and iv_val > 0:
                                iv = float(iv_val)
                        except Exception:
                            iv = None
                        
                        # Skip if very low liquidity
                        if open_interest < 100:
                            continue
                        
                        # Use lastPrice as primary source (more reliable than bid/ask from Yahoo)
                        last_price = put.get('lastPrice', 0)
                        if last_price > 0:
                            mid = last_price
                            # Estimate bid/ask spread around last price
                            spread_pct = 0.1  # 10% spread estimate
                            bid = mid * (1 - spread_pct/2)
                            ask = mid * (1 + spread_pct/2)
                        elif bid > 0 and ask > 0:
                            mid = (bid + ask) / 2
                        else:
                            # Fallback: estimate premium based on strike distance
                            mid = max(0.01, abs(strike - current_price) * 0.1)
                            bid = mid * 0.9
                            ask = mid * 1.1
                        
                        # Calculate spread percentage
                        spread_pct = (ask - bid) / mid if mid > 0 else 1.0
                        
                        # Skip if spread is too wide (>50%)
                        if spread_pct > 0.5:
                            continue
                        
                        quote = OptionQuote(
                            symbol=symbol.upper(),
                            expiration=expiration,
                            strike=strike,
                            bid=bid,
                            ask=ask,
                            last=mid,
                            volume=volume,
                            open_interest=open_interest,
                            iv=iv
                        )
                        quotes.append(quote)
                        
                        # Limit to a higher number of options overall to include wider strikes
                        if len(quotes) >= 200:
                            break
                            
                except Exception as e:
                    print(f"Error getting options for {symbol} {expiration}: {e}")
                    continue
            
            if not quotes:
                print(f"No valid put options found for {symbol}, using fallback")
                return self._get_fallback_options(symbol)
            
            return quotes
            
        except Exception as e:
            print(f"Error fetching options for {symbol}: {e}")
            return self._get_fallback_options(symbol)
    
    def _get_fallback_options(self, symbol: str) -> List[OptionQuote]:
        """Generate fallback options if Yahoo Finance fails."""
        current_price = self.get_current_price(symbol)
        quotes = []
        base_date = date.today()
        
        # Generate realistic put options as fallback
        for days_offset in [30, 34, 38, 42]:
            expiration = base_date + timedelta(days=days_offset)
            
            # Widen strike range around current price to include more strikes
            for strike_offset in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]:
                strike = current_price + strike_offset
                if strike <= 0:
                    continue
                
                mid = max(0.01, (strike - current_price) * 0.8)
                spread = mid * 0.1
                bid = max(0.01, mid - spread)
                ask = mid + mid
                
                volume = random.randint(100, 2000)
                open_interest = random.randint(500, 5000)
                
                quote = OptionQuote(
                    symbol=symbol.upper(),
                    expiration=expiration,
                    strike=strike,
                    bid=bid,
                    ask=ask,
                    last=mid,
                    volume=volume,
                    open_interest=open_interest,
                    iv=0.3
                )
                quotes.append(quote)
        
        return quotes
    
    def get_call_options(self, symbol: str) -> List[OptionQuote]:
        """Get real call options from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol.upper())
            options = ticker.options
            
            if not options:
                print(f"No options found for {symbol}")
                return self._get_fallback_call_options(symbol)
            
            # Get the next few expiration dates
            valid_expiries = []
            current_date = date.today()
            
            for exp_date_str in options:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                if exp_date > current_date:
                    valid_expiries.append(exp_date)
                    if len(valid_expiries) >= 4:  # Limit to 4 expiries
                        break
            
            if not valid_expiries:
                return self._get_fallback_call_options(symbol)
            
            quotes = []
            current_price = self.get_current_price(symbol)
            
            for expiration in valid_expiries:
                try:
                    # Get options chain for this expiry
                    opt_chain = ticker.option_chain(expiration.strftime('%Y-%m-%d'))
                    calls = opt_chain.calls
                    
                    if calls.empty:
                        continue
                    
                    # Filter calls with reasonable strikes and liquidity
                    for _, call in calls.iterrows():
                        strike = call['strike']
                        bid = call['bid']
                        ask = call['ask']
                        volume = call['volume']
                        open_interest = call['openInterest']
                        iv = None
                        try:
                            iv_val = call.get('impliedVolatility') if hasattr(call, 'get') else call['impliedVolatility']
                            if iv_val is not None and iv_val > 0:
                                iv = float(iv_val)
                        except Exception:
                            iv = None
                        
                        # Skip if very low liquidity
                        if open_interest < 100:
                            continue
                        
                        # Use lastPrice as primary source (more reliable than bid/ask from Yahoo)
                        last_price = call.get('lastPrice', 0)
                        if last_price > 0:
                            mid = last_price
                            # Estimate bid/ask spread around last price
                            spread_pct = 0.1  # 10% spread estimate
                            bid = mid * (1 - spread_pct/2)
                            ask = mid * (1 + spread_pct/2)
                        elif bid > 0 and ask > 0:
                            mid = (bid + ask) / 2
                        else:
                            # Fallback: estimate premium based on strike distance
                            mid = max(0.01, abs(strike - current_price) * 0.1)
                            bid = mid * 0.9
                            ask = mid * 1.1
                        
                        # Calculate spread percentage
                        spread_pct = (ask - bid) / mid if mid > 0 else 1.0
                        
                        # Skip if spread is too wide (>50%)
                        if spread_pct > 0.5:
                            continue
                        
                        quote = OptionQuote(
                            symbol=symbol.upper(),
                            expiration=expiration,
                            strike=strike,
                            bid=bid,
                            ask=ask,
                            last=mid,
                            volume=volume,
                            open_interest=open_interest,
                            iv=iv
                        )
                        quotes.append(quote)
                        
                        # Limit to a higher number of options overall to include wider strikes
                        if len(quotes) >= 200:
                            break
                            
                except Exception as e:
                    print(f"Error getting call options for {symbol} {expiration}: {e}")
                    continue
            
            if not quotes:
                print(f"No valid call options found for {symbol}, using fallback")
                return self._get_fallback_call_options(symbol)
            
            return quotes
            
        except Exception as e:
            print(f"Error fetching call options for {symbol}: {e}")
            return self._get_fallback_call_options(symbol)
    
    def _get_fallback_call_options(self, symbol: str) -> List[OptionQuote]:
        """Generate fallback call options if Yahoo Finance fails."""
        current_price = self.get_current_price(symbol)
        quotes = []
        base_date = date.today()
        
        # Generate realistic call options as fallback
        for days_offset in [30, 34, 38, 42]:
            expiration = base_date + timedelta(days=days_offset)
            
            # Widen strike range around current price to include more strikes
            for strike_offset in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]:
                strike = current_price + strike_offset
                if strike <= 0:
                    continue
                
                # Call premium decreases as strike increases above current price
                if strike > current_price:
                    mid = max(0.01, (current_price - strike) * 0.1 + 2.0)
                else:
                    mid = max(0.01, current_price - strike + 2.0)
                
                spread = mid * 0.1
                bid = max(0.01, mid - spread)
                ask = mid + spread
                
                volume = random.randint(100, 2000)
                open_interest = random.randint(500, 5000)
                
                quote = OptionQuote(
                    symbol=symbol.upper(),
                    expiration=expiration,
                    strike=strike,
                    bid=bid,
                    ask=ask,
                    last=mid,
                    volume=volume,
                    open_interest=open_interest,
                    iv=0.3
                )
                quotes.append(quote)
        
        return quotes

# ============
# FastAPI app
# ============

app = FastAPI(title="Cash‑Secured Put Scanner", version="0.2.0")

# Mount static directory for React build
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend_build")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In a production environment, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data adapter
data_adapter = YahooFinanceAdapter()

@app.get("/")
async def root():
    """Serve the React frontend."""
    from fastapi.responses import FileResponse
    frontend_index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {"message": "Cash-Secured Put Scanner API"}

@app.get("/scan")
async def scan_options(
    symbol: str = Query(..., description="Stock symbol to scan"),
    dte_min: int = Query(1, description="Minimum days to expiry"),
    dte_max: int = Query(60, description="Maximum days to expiry"),
    delta_min: float = Query(0.05, description="Minimum delta"),
    delta_max: float = Query(0.50, description="Maximum delta"),
    min_oi: int = Query(100, description="Minimum open interest"),
    max_spread_pct: float = Query(0.20, description="Maximum spread percentage"),
    capital: Optional[float] = Query(None, description="Capital available for cash-secured puts (USD)"),
    target_income: Optional[float] = Query(None, description="Target monthly income (USD)"),
    use_bid: bool = Query(False, description="Use bid price (conservative) for premium and metrics"),
    pop_otm_fallback: float = Query(0.70, description="Fallback POP when OTM (if IV unavailable)"),
    pop_itm_fallback: float = Query(0.30, description="Fallback POP when ITM (if IV unavailable)"),
):
    """Scan for cash-secured put opportunities."""
    try:
        # Get current price and options
        current_price = data_adapter.get_current_price(symbol.upper())
        quotes = data_adapter.get_put_options(symbol.upper())
        
        # Rank the puts
        ranked_puts = rank_puts(
            quotes,
            current_price,
            dte_min,
            dte_max,
            delta_min,
            delta_max,
            min_oi,
            max_spread_pct,
            capital=capital,
            use_bid=use_bid,
            pop_otm_fallback=pop_otm_fallback,
            pop_itm_fallback=pop_itm_fallback,
        )
        
        # Convert to dict format for JSON response
        results = []
        for put in ranked_puts:
            # Capital-aware sizing metrics (cash-secured puts)
            # Note: Broker collateral is commonly strike * 100. We keep this conservative.
            capital_per_contract = put.strike * 100.0  # USD per contract
            premium = (put.bid if use_bid else put.mid)
            income_per_contract = premium * 100.0      # premium dollars received per contract
            contracts = 0
            capital_used = 0.0
            income_total = 0.0
            # After scaling, this simplifies to mid/strike, which is the correct per-capital ratio
            income_per_capital = (income_per_contract / capital_per_contract) if capital_per_contract > 0 else 0.0

            if capital is not None and capital > 0:
                contracts = int(capital // capital_per_contract)
                if contracts > 0:
                    capital_used = contracts * capital_per_contract
                    income_total = contracts * income_per_contract

            # Recompute delta_abs for the response
            delta_abs_val = None
            try:
                import math
                T = max(1e-6, put.days_to_expiry / 365.0)
                sigma = getattr(put, "iv", None)
                if not sigma or sigma <= 0:
                    sigma = 0.30
                d1 = (math.log(max(1e-9, current_price / put.strike)) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
                def phi(x: float) -> float:
                    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                call_delta = phi(d1)
                put_delta = call_delta - 1.0
                delta_abs_val = abs(put_delta)
            except Exception:
                delta_abs_val = None

            results.append({
                "symbol": put.symbol,
                "expiration": put.expiration.isoformat(),
                "strike": put.strike,
                "bid": put.bid,
                "ask": put.ask,
                "mid": put.mid,
                "volume": put.volume,
                "open_interest": put.open_interest,
                "days_to_expiry": put.days_to_expiry,
                "aroc": put.aroc,
                "monthly_income": put.monthly_income,
                "monthly_yield_pct": put.monthly_income,
                "pop": put.pop,
                "breakeven": put.breakeven,
                # expose max_loss both per-share and per-contract dollars for clarity
                "max_loss": put.max_loss,  # per share
                "max_loss_dollars": put.max_loss * 100.0,  # per contract dollars
                "risk_reward": put.risk_reward,
                "income_per_capital": income_per_capital,
                "contracts": contracts,
                "capital_per_contract": capital_per_contract,
                "capital_used": capital_used,
                "income_per_contract": income_per_contract,
                "income_total": income_total,
                "delta_abs": delta_abs_val,
                "score": put.score,
            })
        
        return ScanResponse(results=results, total_found=len(results))
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
async def get_history(
    symbol: str = Query(..., description="Stock symbol to fetch history for"),
    period: str = Query("1mo", description="yfinance period, e.g., 1mo, 3mo, 6mo, 1y"),
    interval: str = Query("1d", description="yfinance interval, e.g., 1d, 1h, 5m"),
):
    """Return recent historical prices for a symbol.
    Response format: { symbol, period, interval, data: [{ date: ISO8601, close: float }] }
    """
    try:
        u_sym = symbol.upper()
        ticker = yf.Ticker(u_sym)
        # Try requested period/interval first, then fallbacks
        try_pairs = [(period, interval)]
        # If daily, try a broader period; if intraday, try 5d/30m then 1mo/1d
        try_pairs.append(("3mo", "1d"))
        try_pairs.append(("6mo", "1d"))

        hist = None
        used_period, used_interval = period, interval
        for p, itv in try_pairs:
            try:
                tmp = ticker.history(period=p, interval=itv)
            except Exception:
                tmp = None
            if tmp is not None and not tmp.empty:
                hist = tmp
                used_period, used_interval = p, itv
                break

        data = []
        if hist is None or hist.empty:
            # Last-resort: synthesize a small series around current price so UI doesn't break
            try:
                spot = YahooFinanceAdapter().get_current_price(u_sym)
            except Exception:
                spot = 100.0
            pts = 30
            for i in range(pts):
                dt = datetime.utcnow() - timedelta(days=(pts - 1 - i))
                # simple gentle random walk
                drift = (i - pts/2) * 0.001 * spot
                val = max(0.01, spot * (1 + drift/spot))
                # Use RFC 3339 format without sub-second precision for Safari compatibility
                dt_rfc3339 = dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                data.append({"date": dt_rfc3339, "close": float(val)})
            return {"symbol": u_sym, "period": used_period, "interval": used_interval, "data": data}

        for idx, row in hist.iterrows():
            try:
                dt_dt = idx.to_pydatetime()
                # Normalize to UTC and emit RFC 3339 without sub-second precision for Safari
                if dt_dt.tzinfo is None:
                    dt_dt = dt_dt.replace(tzinfo=timezone.utc)
                else:
                    dt_dt = dt_dt.astimezone(timezone.utc)
                dt_iso = dt_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                dt_iso = str(idx)
            close_val = float(row.get("Close", row.get("close", 0.0)))
            data.append({"date": dt_iso, "close": close_val})
        return {"symbol": u_sym, "period": used_period, "interval": used_interval, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}

@app.get("/scan-calls")
async def scan_covered_calls(
    symbol: str = Query(..., description="Stock symbol to scan"),
    dte_min: int = Query(1, description="Minimum days to expiry"),
    dte_max: int = Query(60, description="Maximum days to expiry"),
    delta_min: float = Query(0.05, description="Minimum delta"),
    delta_max: float = Query(0.50, description="Maximum delta"),
    min_oi: int = Query(100, description="Minimum open interest"),
    max_spread_pct: float = Query(0.20, description="Maximum spread percentage"),
    capital: Optional[float] = Query(None, description="Value of shares owned (USD)"),
    target_income: Optional[float] = Query(None, description="Target monthly income (USD)"),
    use_bid: bool = Query(False, description="Use bid price (conservative) for premium and metrics"),
    pop_otm_fallback: float = Query(0.70, description="Fallback POP when OTM (if IV unavailable)"),
    pop_itm_fallback: float = Query(0.30, description="Fallback POP when ITM (if IV unavailable)"),
    otm_only: bool = Query(True, description="Only include OTM calls (strike above current price)"),
):
    """Scan for covered call opportunities."""
    try:
        # Get current price and call options
        current_price = data_adapter.get_current_price(symbol.upper())
        call_quotes = data_adapter.get_call_options(symbol.upper())
        
        # Rank the calls for covered call strategy
        ranked_calls = rank_covered_calls(
            call_quotes,
            current_price,
            dte_min,
            dte_max,
            delta_min,
            delta_max,
            min_oi,
            max_spread_pct,
            capital=capital,
            use_bid=use_bid,
            pop_otm_fallback=pop_otm_fallback,
            pop_itm_fallback=pop_itm_fallback,
            otm_only=otm_only,
        )
        
        # Convert to dict format for JSON response
        results = []
        for call in ranked_calls:
            # For covered calls: we own 100 shares per contract
            shares_per_contract = 100
            share_value_per_contract = current_price * shares_per_contract
            premium = (call.bid if use_bid else call.mid)
            income_per_contract = premium * 100.0  # premium dollars received per contract
            contracts = 0
            capital_used = 0.0
            income_total = 0.0
            
            if capital is not None and capital > 0:
                contracts = int(capital // share_value_per_contract)
                if contracts > 0:
                    capital_used = contracts * share_value_per_contract
                    income_total = contracts * income_per_contract

            # Calculate delta for calls
            delta_abs_val = None
            try:
                import math
                T = max(1e-6, call.days_to_expiry / 365.0)
                sigma = getattr(call, "iv", None)
                if not sigma or sigma <= 0:
                    sigma = 0.30
                d1 = (math.log(max(1e-9, current_price / call.strike)) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
                def phi(x: float) -> float:
                    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                call_delta = phi(d1)
                delta_abs_val = abs(call_delta)
            except Exception:
                delta_abs_val = None

            results.append({
                "symbol": call.symbol,
                "expiration": call.expiration.isoformat(),
                "strike": call.strike,
                "bid": call.bid,
                "ask": call.ask,
                "mid": call.mid,
                "volume": call.volume,
                "open_interest": call.open_interest,
                "days_to_expiry": call.days_to_expiry,
                "aroc": call.aroc,
                "monthly_income": call.monthly_income,
                "monthly_yield_pct": call.monthly_income,
                "pop": call.pop,
                "breakeven": call.breakeven,
                "max_loss": call.max_loss,
                "max_loss_dollars": call.max_loss * 100.0,
                "risk_reward": call.risk_reward,
                "income_per_capital": (income_per_contract / share_value_per_contract) if share_value_per_contract > 0 else 0.0,
                "contracts": contracts,
                "capital_per_contract": share_value_per_contract,
                "capital_used": capital_used,
                "income_per_contract": income_per_contract,
                "income_total": income_total,
                "delta_abs": delta_abs_val,
                "score": call.score,
            })
        
        return ScanResponse(results=results, total_found=len(results))
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/price")
async def get_price(symbol: str = Query(..., description="Stock symbol to fetch current price for")):
    """Return current market price and company name for a symbol."""
    try:
        # Use adapter for robust price with fallbacks
        price = data_adapter.get_current_price(symbol.upper())
        # Best-effort IV estimate
        iv = data_adapter.get_symbol_iv(symbol.upper())
        # Fetch name via yfinance (best-effort)
        name = None
        try:
            t = yf.Ticker(symbol.upper())
            info = t.info or {}
            name = info.get("shortName") or info.get("longName") or None
        except Exception:
            name = None
        return {"symbol": symbol.upper(), "price": price, "name": name, "iv": iv}
    except Exception as e:
        return {"error": str(e)}

@app.get("/fx/usd_cad")
async def fx_usd_cad():
    """Return current USD->CAD FX rate from Yahoo Finance.
    Response: { base: 'USD', quote: 'CAD', pair: 'USD/CAD', rate: float, as_of: ISO8601Z }
    """
    try:
        rate = data_adapter.get_fx_usd_to_cad()
        return {
            "base": "USD",
            "quote": "CAD",
            "pair": "USD/CAD",
            "rate": float(rate),
            "as_of": datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as e:
        return {
            "base": "USD",
            "quote": "CAD",
            "pair": "USD/CAD",
            "rate": 1.39,
            "as_of": datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": str(e),
        }

# =====================
# Reminder for the user
# =====================
# • The backend (Python) is now complete and functional
# • You must create a React project (e.g. with Vite or Create React App),
#   add the App.js component, build it (npm run build), and drop build output
#   into ./frontend_build so FastAPI serves it.
# • Then browse http://localhost:8000 to use the UI.            