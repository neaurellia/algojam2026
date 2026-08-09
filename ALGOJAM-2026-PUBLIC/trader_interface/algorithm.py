"""Combined book: catherine's refined core plus hhanyu615's new instruments.

Where both authors cover an instrument, the version with the smaller fit/test
imbalance wins even when it earns less, because the Round 1 headline is not
what gets scored:

  UQ Dollar        neither. Both books hardcoded a $100 anchor; this one uses
                   a 90-day rolling mean instead. Earns $3,237 less on Round 1
                   and removes the only hardcoded LEVEL in the file -- see the
                   UQ Dollar block below for why that trade is worth making.
  Sausage Sizzle   neither. Both books hardcoded coefficients from an OLS fit
                   over all 365 days and applied them from day 0, so the
                   fit/test split never actually tested them. This one refits
                   daily on past data only -- see the Sausage Sizzle block.
  Boat Party       catherine's window ENSEMBLE over hhanyu615's w=5. The single
                   window earns more ($59,050 vs $43,320) but was chosen by
                   looking at results; the vote needs no choice at all.
  Fintech Token    catherine's w=5 plus the volatility filter, measured at
                   $91,724 against $88,422 without it.
  Thrifted Jeans   catherine's (3,15,0.05) over hhanyu615's (5,20,0.02). The
                   latter earns $81,664 but is the single best of 45 grid
                   combinations, with a 24.5x fit/test ratio; the former is
                   $63,816 at 1.7x.
  Bread            catherine's k=7 momentum over hhanyu615's Donchian w=8.
                   Donchian earns more ($26,985 vs $22,090) but splits
                   $18,595/$8,390 across the halves against a near-perfect
                   $11,070/$11,020. Set BREAD_DONCHIAN = True to compare.
  Sausage          hhanyu615's Donchian w=8, new.
  MenuDash         hhanyu615's z-score w=10, new. Far better than the static
                   long this book previously assumed ($67,500 vs $12,750).

Eight instruments do not fit in the $600K daily budget at full size --
hhanyu615's book peaks at $603,291, which would have the engine zero every
position on that day. enforce_budget below is what makes the combination
viable.
"""
import math

import numpy as np


class Algorithm():

    # Instruments this algorithm takes a position on. Anything not listed here
    # is held flat.
    ENABLED = [
        "UQ Dollar",
        "Sausage Sizzle",
        "Boat Party Ticket",
        "Fintech Token",
        "Thrifted Jeans",
        "Sausage",
        "Bread",
        "MenuDash",
    ]

    # Order in which instruments are kept when the daily budget gets tight:
    # last entries are dropped first.
    #
    # Ranked by how much each signal can be relied on, then by how much budget
    # it frees per dollar of P&L given up. Sizzle and UQ Dollar have the two
    # highest captures and the evenest halves. MenuDash is dropped early
    # despite earning well because it is the most expensive thing here
    # (~$98K average exposure for 13.7% capture), so it frees the most room
    # per dollar sacrificed. Thrifted Jeans is next because it is the only
    # signal that fails a shuffle test. Liferaft is last: one unit is ~$150K
    # and it pays no local P&L at all.
    PRIORITY = [
        "Sausage Sizzle",
        "UQ Dollar",
        "Fintech Token",
        "Boat Party Ticket",
        "Bread",
        "Sausage",
        "Thrifted Jeans",
        "MenuDash",
        "Liferaft Ticket",
    ]

    # Stay this far under the exchange's $600K cap. The engine values positions
    # at the same prices used here, so without a margin a rounding edge could
    # tip the total over -- and a breach zeroes EVERY position for the day, not
    # just the offending one.
    BUDGET_CEILING = 580_000

    # ---- UQ Dollar ----------------------------------------------------------
    # Hard $100 peg: ACF(1) of daily changes = -0.49, half-life 0.7 days.
    #
    # The anchor is a rolling mean, NOT the constant 100.0 this used to hold.
    # That swap trades a fitted LEVEL for a fitted WINDOW, and the two fail very
    # differently. Every other signal here is scale-free -- differences, means
    # minus price, z-scores, breakouts -- so all of them survive a Round 2 that
    # re-prices. A hardcoded level does not: if the peg sits anywhere but $100
    # we hold +/-650 units the wrong way EVERY day, on ~$63K of exposure, and
    # the most dependable line in the book becomes the largest loser.
    #
    # The window is benign where the level was not. Measured on Round 1, every
    # setting from 45 to 150 lands between $60,210 and $65,488 -- the choice
    # barely matters. 90 is mid-plateau, deliberately not the $65,488 peak at
    # w=45: picking the best cell of a grid is what this change exists to avoid.
    #
    # Costs $3,237 against the constant's $64,552 (5% of the line, 0.8% of the
    # book) and is slightly BETTER balanced across the halves, 1.06 vs 1.11.
    # That is the insurance premium for removing the only single-point failure
    # in the file.
    #
    # No warmup guard and no fallback to 100.0 on purpose. Slicing [-90:] on
    # short history just yields an expanding mean, so the anchor self-calibrates
    # from day 0 -- on day 0 it equals today's price, giving signal 0, i.e.
    # flat, which is the correct view when there is no history. A constant
    # fallback would reintroduce the exact level bet being removed here.
    UQ_ANCHOR_WINDOW = 90

    # ---- Boat Party Ticket --------------------------------------------------
    # Averaged across four windows rather than committing to one. The parameter
    # plateau is badly shaped -- positive at only 8 of 11 windows, swinging
    # $100K across the range -- so any single choice is a guess.
    BOAT_WINDOWS = (3, 5, 10, 20)

    # ---- Fintech Token ------------------------------------------------------
    # Single window: the plateau is positive at 11 of 11 settings with w=5
    # mid-plateau, so there is nothing to average away.
    FINTECH_WINDOW = 5

    # Volatility filter: go flat after an outsized single-day move, resume once
    # calm. MEASURED, and it earns its place -- but NOT for the reason it was
    # built. The theory was that it would rescue the days 120-150 crash. It
    # does not: that window loses $16,945 either way and none of the 18 days it
    # pauses fall inside it. It sits out three other volatility clusters
    # instead, worth +$6,395 on the fitted half and -$3,093 on the held-out
    # half. Treat as provisional: a change that works for different reasons
    # than predicted is not yet understood.
    FT_JUMP_PAUSE = True
    FT_JUMP_SIGMA = 3.0        # a move over N rolling std devs starts a pause
    FT_CALM_DAYS = 2           # consecutive quiet days needed to resume
    FT_VOL_WINDOW = 20

    # ---- Sausage Sizzle -----------------------------------------------------
    # Sizzle re-prices off the PREVIOUS day's moves in its inputs: d(Sausage)
    # leads d(Sizzle) by one day at corr +0.61 and d(Bread) at +0.71, while
    # contemporaneous correlation is ~0.02. It is a cost passthrough, not a
    # chart pattern, which is why it generalises at 90.6% capture.
    #
    # The coefficients are now REFIT EVERY DAY on history up to that day, where
    # they used to be the three constants below -- an OLS fit over all 365 days,
    # applied from day 0. That fit was never wrong, but it was never TESTED
    # either: both halves of the fit/test split were in-sample for it, so
    # Sizzle's 90.6% capture was the one number in this book the split did not
    # validate. Refitting causally makes the reported number mean what every
    # other line's number means.
    #
    # Was expanding (SIZZLE_WINDOW = None) on the reasoning that a rolling
    # window adds a fitted LENGTH -- the same kind of choice the UQ Dollar
    # block above exists to avoid. Moved to a 60-day trailing window instead:
    # the Bread/Sausage relationship drifts over the year, and a fixed window
    # lets the daily refit track the current regime instead of being
    # increasingly dominated by however many days have piled up. Set
    # SIZZLE_WINDOW back to None to compare the expanding fit.
    #
    # Only the SIGN of the prediction is traded, so what matters is the ratio
    # between the coefficients, not their scale. That is why this survives the
    # early days when the estimates are still noisy: the ratio settles long
    # before the magnitudes do.
    SIZZLE_ROLLING = True
    SIZZLE_WINDOW = 60         # None = expanding; int = trailing window
    SIZZLE_MIN_SAMPLE = 20     # stay flat until the fit has this many days
    # const/Bread/Sausage below are retained only as the comparison case,
    # used when SIZZLE_ROLLING is False. Those three are the full-sample fit
    # and are NOT causal.
    #
    # MenuDash is different: it is NOT refit daily (there is no causal
    # rolling equivalent below), so its coefficient and halflife are always
    # the ones traded, in both branches -- see sizzle_menudash_term. Raw
    # MenuDash adds almost nothing to this model (R2 0.8511 -> 0.8560): the
    # posted daily number is mostly the app's rounding and surge noise.
    # SMOOTHED MenuDash adds real signal (-> 0.8856), because the EWMA tracks
    # the underlying labour cost that actually feeds Sizzle's price. Every
    # halflife from 2 to 20 improves BOTH halves, so this is a plateau rather
    # than a peak. Worth +$2,520 and +2.5pp directional accuracy (82.6->85.1%)
    # measured against the full-sample Bread/Sausage fit above; not
    # re-measured against the 60-day rolling refit it is now added on top of.
    SIZZLE_COEFFS = {"const": 0.0053, "Bread": 0.0769,
                     "Sausage": 1.6780, "MenuDash": 4.2432}
    SIZZLE_MENUDASH_HALFLIFE = 5

    # ---- Thrifted Jeans -----------------------------------------------------
    # MA crossover, traded only while the slow MA is actually going somewhere.
    # Across a 45-combination grid every setting was positive overall, but the
    # median first half earned -$688: the edge exists only in trending
    # stretches. This is the most selective setting tested and the only one
    # with a fit/test ratio near 1.
    TJ_FAST_WINDOW = 3
    TJ_SLOW_WINDOW = 15
    TJ_SLOPE_LOOKBACK = 10     # compare the slow MA against N days ago
    TJ_MIN_SLOPE = 0.05        # need >5% change over that span to trade

    # ---- Sausage ------------------------------------------------------------
    # Donchian breakout. Same-length move autocorrelation peaks at +0.324 for
    # 5d and 8d -- the highest of any instrument here -- then reverses hard at
    # 20d (-0.354) and 30d (-0.702). A slope filter was tested and HURT badly
    # ($1,950 vs $10,500): unlike Thrifted Jeans, Sausage has no sustained
    # sideways ranges, so filtering only keeps us out of live trends.
    SAUSAGE_WINDOW = 8

    # ---- Bread --------------------------------------------------------------
    # Momentum against a week ago. hhanyu615's Donchian w=8 earns more
    # ($26,985 vs $22,090) but splits $18,595/$8,390 across the halves against
    # this one's $11,070/$11,020 -- the evenest split in either book. Flip
    # BREAD_DONCHIAN to compare them directly.
    BREAD_LOOKBACK = 7
    BREAD_DONCHIAN = False
    BREAD_WINDOW = 8

    # ---- MenuDash -----------------------------------------------------------
    # MenuDash posts a noisy read on a smooth underlying cost. The noise is
    # small and frequent (1-3 day wobbles) rather than large and rare, so a
    # short-window z-score with a deadband captures it.
    MENUDASH_WINDOW = 10
    MENUDASH_THRESHOLD = 0.5

    # Human-readable description of each signal, keyed by instrument. report.py
    # prints this straight out, so the documented settings can never drift from
    # the ones actually traded. "params" counts values fitted to the data.
    SPEC = {
        "UQ Dollar":         ("w = 90",    "mean(P[-90:]) - P[-1]",          1),
        "Sausage Sizzle":    ("daily refit, w=60", "OLS on Bread/Sausage + smoothed MenuDash, sign traded", 3),
        "Boat Party Ticket": ("ensemble",  "sign vote, w in 3/5/10/20",      0),
        "Fintech Token":     ("w = 5",     "mean(P[-5:]) - P[-1] + vol gate", 1),
        "Thrifted Jeans":    ("MA 3/15",   "fast-slow, slope>5% over 10d",   4),
        "Sausage":           ("w = 8",     "Donchian breakout, held between", 1),
        "Bread":             ("k = 7",     "P[-1] - P[-8]  (momentum)",      1),
        "MenuDash":          ("w = 10",    "-(z-score), |z| > 0.5",          2),
    }

    # FUNCTION TO SETUP ALGORITHM CLASS
    def __init__(self, positions):
        self.data = {}  # Historical data of all instruments
        self.positionLimits = {}    # Initialise position limits
        self.day = 0     # Initialise the current day as 0
        self.positions = positions   # Initialise the current positions
        # Fintech volatility-filter state, and the Donchian directions, all of
        # which persist across days because the engine reuses this object.
        self.ft_paused = False
        self.ft_calm_run = 0
        self.sausage_direction = 0
        self.bread_direction = 0

    def get_current_price(self, instrument):
        """
        Helper function to fetch current price of an instrument.
        """
        return self.data[instrument][-1]

    def reversion_signal(self, instrument, window):
        """
        Positive when price sits below its recent mean, i.e. when we expect it
        to rise. Returns None until there is enough history.
        """
        prices = self.data[instrument]
        if len(prices) < window:
            return None
        recent = np.array(prices[-window:], dtype=float)
        return recent.mean() - recent[-1]

    def ensemble_reversion_signal(self, instrument, windows):
        """
        Majority vote of the reversion signal across several windows. Each
        window contributes only its DIRECTION, not its magnitude.

        Averaging the raw signals instead would quietly reweight the vote: a
        20-day mean sits much further from today's price than a 3-day mean, so
        the longest window would dominate and the ensemble would just be a
        slow signal wearing a disguise. Voting keeps every horizon equal.
        """
        votes = []
        for window in windows:
            signal = self.reversion_signal(instrument, window)
            if signal is not None:
                votes.append(np.sign(signal))
        if not votes:
            return None
        return float(sum(votes) / len(votes))

    def donchian_signal(self, instrument, window, attribute):
        """
        Breakout: a new N-day high flips us long, a new N-day low flips us
        short, and between the two extremes we HOLD the last direction rather
        than re-deciding, which is what stops it churning on noise. Returns 0
        until the first breakout, i.e. flat.
        """
        prices = self.data[instrument]
        if len(prices) < window:
            return None
        recent = np.array(prices[-window:], dtype=float)
        if prices[-1] >= recent.max():
            setattr(self, attribute, 1)
        elif prices[-1] <= recent.min():
            setattr(self, attribute, -1)
        return float(getattr(self, attribute))

    def uq_signal(self):
        """
        Positive when UQ Dollar sits below its own rolling mean, i.e. when the
        peg should pull it back up.

        Deliberately NOT reversion_signal, which returns None until its window
        fills and would sit flat for the first 90 days. Slicing a short list
        just yields an expanding mean, so the anchor is usable from day 0 and
        never depends on a hardcoded level.
        """
        recent = np.array(self.data["UQ Dollar"][-self.UQ_ANCHOR_WINDOW:],
                          dtype=float)
        return recent.mean() - recent[-1]

    def sizzle_inputs(self):
        """Today's moves in the two cost inputs, or None if unavailable."""
        moves = []
        for name in ("Bread", "Sausage"):
            prices = self.data.get(name)
            if prices is None or len(prices) < 2:
                return None
            moves.append(prices[-1] - prices[-2])
        return moves

    def sizzle_coefficients(self):
        """
        Least-squares fit of tomorrow's Sizzle move on today's input moves,
        using only days that have already happened.

        Training pairs are (input moves on day k, Sizzle move on day k+1) for
        every k where both sides are known. Today's own input move is
        deliberately NOT a training row: the Sizzle move it predicts has not
        happened yet, and including it is precisely the one-day lookahead this
        method exists to remove.

        Returns None until SIZZLE_MIN_SAMPLE pairs exist, or if the inputs are
        collinear enough that the fit is not defined.
        """
        series = {}
        for name in ("Sausage Sizzle", "Bread", "Sausage"):
            prices = self.data.get(name)
            if prices is None or len(prices) < 3:
                return None
            series[name] = np.diff(np.asarray(prices, dtype=float))

        # diff[i] is the move INTO day i+1, so dropping the last input row and
        # the first target row lines each input move up with the Sizzle move
        # one day after it.
        targets = series["Sausage Sizzle"][1:]
        features = np.column_stack([
            series["Bread"][:-1],
            series["Sausage"][:-1],
            np.ones(len(targets)),
        ])

        if self.SIZZLE_WINDOW is not None:
            targets = targets[-self.SIZZLE_WINDOW:]
            features = features[-self.SIZZLE_WINDOW:]

        if len(targets) < self.SIZZLE_MIN_SAMPLE:
            return None

        solution, _, rank, _ = np.linalg.lstsq(features, targets, rcond=None)
        if rank < features.shape[1]:
            # Underdetermined: lstsq still returns a vector, but it is the
            # minimum-norm one rather than a fit, so it says nothing.
            return None
        return solution

    def sizzle_menudash_term(self):
        """
        Smoothed-MenuDash contribution, added on top of the Bread/Sausage
        prediction in sizzle_signal.

        MenuDash enters SMOOTHED, not raw -- see the SIZZLE_COEFFS note above.
        The EWMA is recomputed over full history each call (cheap at this
        series length) and only its one-day CHANGE is used, so this returns
        0.0 rather than None when history is short: it is a bolt-on term, not
        a gate on the whole signal.
        """
        menudash = self.data.get("MenuDash")
        if menudash is None or len(menudash) < 2:
            return 0.0
        alpha = 1 - 0.5 ** (1.0 / self.SIZZLE_MENUDASH_HALFLIFE)
        ewma = menudash[0]
        previous = ewma
        for price in menudash[1:]:
            previous = ewma
            ewma = alpha * price + (1 - alpha) * ewma
        return self.SIZZLE_COEFFS["MenuDash"] * (ewma - previous)

    def sizzle_signal(self):
        """
        Predict tomorrow's move in Sausage Sizzle from today's moves in Bread
        and Sausage, plus a smoothed-MenuDash term.

        The Bread/Sausage coefficients come from a daily refit on a trailing
        SIZZLE_WINDOW-day window of past data. Returns None while the fit is
        still too short to trust, which keeps us flat rather than falling
        back on the static SIZZLE_COEFFS -- reverting to those would
        reintroduce the full-sample fit being removed here, exactly as a
        constant fallback would for the UQ Dollar anchor.

        MenuDash is NOT part of that refit -- it is added via the fixed
        SIZZLE_COEFFS["MenuDash"] coefficient in both branches, since there is
        no causal rolling equivalent for it here.
        """
        moves = self.sizzle_inputs()
        if moves is None:
            return None
        d_bread, d_sausage = moves

        if not self.SIZZLE_ROLLING:
            prediction = (self.SIZZLE_COEFFS["const"]
                          + self.SIZZLE_COEFFS["Bread"] * d_bread
                          + self.SIZZLE_COEFFS["Sausage"] * d_sausage)
        else:
            coefficients = self.sizzle_coefficients()
            if coefficients is None:
                return None
            b_bread, b_sausage, const = coefficients
            prediction = const + b_bread * d_bread + b_sausage * d_sausage

        return prediction + self.sizzle_menudash_term()

    def jeans_signal(self):
        """
        MA crossover gated by the slow MA's slope. Returns the fast-slow gap
        while the slow MA is trending, 0 while it is flat (stay out of the
        chop), and None until there is enough history for both.
        """
        prices = self.data["Thrifted Jeans"]
        span = self.TJ_SLOW_WINDOW + self.TJ_SLOPE_LOOKBACK
        if len(prices) < span:
            return None

        recent = np.array(prices, dtype=float)
        fast = recent[-self.TJ_FAST_WINDOW:].mean()
        slow = recent[-self.TJ_SLOW_WINDOW:].mean()
        # The slow MA as it stood TJ_SLOPE_LOOKBACK days ago.
        previous_slow = recent[-span:-self.TJ_SLOPE_LOOKBACK].mean()
        if previous_slow == 0:
            return None

        slope = (slow - previous_slow) / previous_slow
        if abs(slope) < self.TJ_MIN_SLOPE:
            return 0.0
        return fast - slow

    def bread_signal(self):
        """Momentum against a week ago, or Donchian if BREAD_DONCHIAN is set."""
        if self.BREAD_DONCHIAN:
            return self.donchian_signal("Bread", self.BREAD_WINDOW,
                                        "bread_direction")
        prices = self.data["Bread"]
        if len(prices) <= self.BREAD_LOOKBACK:
            return None
        return prices[-1] - prices[-(self.BREAD_LOOKBACK + 1)]

    def menudash_signal(self):
        """
        Positive when MenuDash sits below its own recent mean, i.e. when we
        expect the posted price to drift back up. Flat when the deviation is
        too small to be worth trading. Returns None until there is enough
        history.
        """
        prices = self.data["MenuDash"]
        if len(prices) < self.MENUDASH_WINDOW:
            return None
        recent = np.array(prices[-self.MENUDASH_WINDOW:], dtype=float)
        sd = recent.std()
        if sd == 0:
            return None
        z = (recent[-1] - recent.mean()) / sd
        if abs(z) < self.MENUDASH_THRESHOLD:
            return 0.0
        # Above the mean -> expect a fall -> negative signal -> short
        return -z

    def fintech_is_paused(self):
        """
        Track whether an outsized move has put Fintech on the sidelines.

        A move larger than FT_JUMP_SIGMA times its own recent volatility means
        the level has changed, so the rolling mean is now an average of two
        different regimes and cannot be reverted to. Sit out until the price
        has been quiet for FT_CALM_DAYS in a row; any loud day resets that
        count, so a choppy settling period keeps us out until it truly calms.
        """
        prices = self.data["Fintech Token"]
        if len(prices) < self.FT_VOL_WINDOW + 1:
            return False

        recent = np.array(prices[-(self.FT_VOL_WINDOW + 1):], dtype=float)
        moves = np.diff(recent)
        # Exclude today's move from its own volatility estimate, or a big jump
        # inflates the denominator and hides itself.
        volatility = moves[:-1].std()
        if volatility == 0:
            return self.ft_paused

        magnitude = abs(moves[-1]) / volatility
        if magnitude > self.FT_JUMP_SIGMA:
            self.ft_paused = True
            self.ft_calm_run = 0
        elif self.ft_paused:
            if magnitude < 1.0:
                self.ft_calm_run += 1
                if self.ft_calm_run >= self.FT_CALM_DAYS:
                    self.ft_paused = False
                    self.ft_calm_run = 0
            else:
                self.ft_calm_run = 0
        return self.ft_paused

    def get_signal(self, instrument):
        """Signal for one instrument; sign is the direction we want to hold."""
        if instrument == "UQ Dollar":
            return self.uq_signal()
        if instrument == "Sausage Sizzle":
            return self.sizzle_signal()
        if instrument == "Thrifted Jeans":
            return self.jeans_signal()
        if instrument == "Boat Party Ticket":
            return self.ensemble_reversion_signal(instrument, self.BOAT_WINDOWS)
        if instrument == "Sausage":
            return self.donchian_signal("Sausage", self.SAUSAGE_WINDOW,
                                        "sausage_direction")
        if instrument == "Bread":
            return self.bread_signal()
        if instrument == "MenuDash":
            return self.menudash_signal()
        if instrument == "Fintech Token":
            # State has to advance every day, so update the filter before
            # deciding whether to act on it.
            if self.FT_JUMP_PAUSE and self.fintech_is_paused():
                return 0.0
            return self.reversion_signal(instrument, self.FINTECH_WINDOW)
        return None

    def enforce_budget(self, desiredPositions):
        """
        Trim positions, cheapest-to-lose first, until the day's total exposure
        fits under BUDGET_CEILING.

        Walks reverse PRIORITY and SHRINKS each position by just enough units
        to close the gap, only zeroing an instrument when trimming it whole
        still is not enough. Reverse-priority order still protects Sizzle and
        UQ Dollar -- the trimming starts at the least trusted end -- so this
        keeps the old ordering logic and only changes how much gets taken.

        This used to zero whole instruments, which was wildly disproportionate.
        Measured on Round 1: the guard fires on 30 of 365 days, drops MenuDash
        every single time and nothing else ever, and the median day needed
        $8,340 while freeing $147,000 -- an 18.6x overshoot. Day 44 needed $373
        and freed $142,500. MenuDash at ~75,000 units and ~$1.87 only had to
        give up ~6% of its position on a typical firing.

        Trimming instead is worth $11,511 (total $423,294 -> $434,805), and
        essentially all of it lands in the held-out half: FIT moves -$1,465
        while TEST moves +$12,975. That asymmetry is expected rather than
        lucky, because this adds NO parameters -- given that $X must be shed,
        shedding exactly $X rather than 18x $X is not a view about the data.

        The old comment argued against "scaling everything down proportionally"
        and it was right to: that would shrink the best signals to fund the
        worst. This is the different thing it did not consider -- scaling only
        the marginal instrument, the one already chosen to be sacrificed.
        """
        def exposure():
            return sum(abs(position) * self.data[instrument][-1]
                       for instrument, position in desiredPositions.items()
                       if position)

        total = exposure()
        if total <= self.BUDGET_CEILING:
            return desiredPositions

        # PRIORITY first, then anything it forgot. Without that tail an
        # instrument added to ENABLED but missing from PRIORITY could never be
        # trimmed, the loop would end still over budget, and the ENGINE zeroes
        # EVERY position for the day -- far worse than any trim.
        order = list(reversed(self.PRIORITY))
        order += [name for name in desiredPositions if name not in self.PRIORITY]

        actions = []
        for instrument in order:
            if total <= self.BUDGET_CEILING:
                break
            position = desiredPositions.get(instrument)
            if not position:
                continue
            price = self.data[instrument][-1]
            if price <= 0:
                continue
            # ceil, so rounding always lands us UNDER the ceiling, never a
            # dollar over it.
            shed = math.ceil((total - self.BUDGET_CEILING) / price)
            if shed < abs(position):
                keep = abs(position) - shed
                desiredPositions[instrument] = keep if position > 0 else -keep
                actions.append(f"{instrument} {abs(position)}->{keep}")
            else:
                desiredPositions[instrument] = 0
                actions.append(f"{instrument} flat")
            total = exposure()
        print(f"BUDGET GUARD: {', '.join(actions)} -> ${total:,.0f}")
        return desiredPositions

    # RETURN DESIRED POSITIONS IN DICT FORM
    def get_positions(self):
        # Get current position
        currentPositions = self.positions
        # Get position limits
        positionLimits = self.positionLimits

        # Declare a store for desired positions
        desiredPositions = {}
        # Loop through all the instruments you can take positions on.
        for instrument, positionLimit in positionLimits.items():
            # For each instrument initilise desired position to zero
            desiredPositions[instrument] = 0

        # IMPLEMENT CODE HERE TO DECIDE WHAT POSITIONS YOU WANT
        #######################################################################
        # Display the current trading day
        print("Starting Algorithm for Day:", self.day)

        # Greedy sizing: full position limit in whichever direction the signal
        # points, then enforce_budget trims whole instruments if the day does
        # not fit under the cap.
        for instrument in self.ENABLED:
            signal = self.get_signal(instrument)
            if signal is None or signal == 0:
                # Not enough history yet, or no view today. Stay flat.
                continue
            limit = positionLimits[instrument]
            desiredPositions[instrument] = limit if signal > 0 else -limit
            print(f"{instrument}: ${self.get_current_price(instrument)} "
                  f"signal={signal:+.4f} position={desiredPositions[instrument]}")

        desiredPositions = self.enforce_budget(desiredPositions)

        # Display the end of trading day
        print("Ending Algorithm for Day:", self.day, "\n")
        #######################################################################
        # Return the desired positions
        return desiredPositions
